// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use clap::Parser;
use modelexpress_server::{
    backend_config::BackendConfig,
    config::{ServerArgs, ServerConfig},
    run_server,
};
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::{SpanExporter, WithExportConfig};
use opentelemetry_sdk::{Resource, trace::SdkTracerProvider};
use tokio::signal::unix::{SignalKind, signal};
use tracing::{error, info};
use tracing_subscriber::{EnvFilter, layer::SubscriberExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Parse command line arguments
    let args = ServerArgs::parse();

    // Check if we should validate config and exit
    if args.validate_config {
        match ServerConfig::load_and_validate_strict(args) {
            Ok(config) => {
                println!("Configuration is valid ✓");
                config.print_config();
                return Ok(());
            }
            Err(e) => {
                eprintln!("Configuration validation failed: {e}");
                std::process::exit(1);
            }
        }
    }

    // Load configuration from multiple sources
    let config = ServerConfig::load(args)?;

    // Initialize tracing with the configured log level
    let log_level = config.log_level();

    let trace_provider = match std::env::var("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") {
        Ok(endpoint) => {
            // OTLP's HTTP client uses rustls before model providers initialize it.
            let _ = rustls::crypto::ring::default_provider().install_default();
            let exporter = SpanExporter::builder()
                .with_http()
                .with_endpoint(endpoint)
                .build()?;
            Some(
                SdkTracerProvider::builder()
                    .with_batch_exporter(exporter)
                    .with_resource(
                        Resource::builder()
                            .with_service_name("modelexpress-server")
                            .build(),
                    )
                    .build(),
            )
        }
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => return Err(error.into()),
    };
    let otel_layer = trace_provider.as_ref().map(|provider| {
        tracing_opentelemetry::layer().with_tracer(provider.tracer("modelexpress-server"))
    });
    let filter = refit_log_filter(
        log_level,
        trace_provider.is_some(),
        &std::env::var("RUST_LOG").unwrap_or_default(),
    );
    let subscriber = tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer())
        .with(otel_layer);
    tracing::subscriber::set_global_default(subscriber)?;

    // Shut down gracefully on CTRL+C (SIGINT) or SIGTERM. SIGTERM is what
    // Kubernetes and container runtimes send to stop a container; as PID 1 the
    // server would otherwise ignore it and be SIGKILLed after the grace period.
    let mut sigterm = signal(SignalKind::terminate())?;
    let shutdown = async move {
        tokio::select! {
            result = tokio::signal::ctrl_c() => match result {
                Ok(()) => info!("Received CTRL+C, shutting down gracefully..."),
                Err(e) => error!("Failed to install CTRL+C signal handler: {e}"),
            },
            _ = sigterm.recv() => info!("Received SIGTERM, shutting down gracefully..."),
        }
    };

    let backend = BackendConfig::from_env()?;

    let result = run_server(config, backend, shutdown).await;
    if let Some(provider) = trace_provider
        && let Err(error) = provider.shutdown()
    {
        if result.is_ok() {
            return Err(error.into());
        }
        error!("Failed to shut down telemetry: {error}");
    }
    result
}

fn refit_log_filter(
    log_level: tracing::Level,
    telemetry_enabled: bool,
    overrides: &str,
) -> EnvFilter {
    let mut directives = log_level.to_string();
    if telemetry_enabled {
        directives.push_str(
            ",modelexpress_server::telemetry=info,modelexpress_server::refit::service=info",
        );
    }
    directives.push(',');
    directives.push_str(overrides);
    EnvFilter::new(directives)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_log_directives_override_server_and_telemetry_defaults() {
        for (telemetry, overrides, general_debug, telemetry_info, service_info) in [
            (false, "", false, false, false),
            (false, "debug", true, true, true),
            (true, "", false, true, true),
            (
                true,
                "debug,modelexpress_server::telemetry=off,modelexpress_server::refit::service=error",
                true,
                false,
                false,
            ),
        ] {
            let subscriber = tracing_subscriber::registry().with(refit_log_filter(
                tracing::Level::WARN,
                telemetry,
                overrides,
            ));
            tracing::subscriber::with_default(subscriber, || {
                assert_eq!(
                    tracing::enabled!(target: "modelexpress_server::server", tracing::Level::DEBUG),
                    general_debug
                );
                assert_eq!(
                    tracing::enabled!(target: "modelexpress_server::telemetry", tracing::Level::INFO),
                    telemetry_info
                );
                assert_eq!(
                    tracing::enabled!(target: "modelexpress_server::refit::service", tracing::Level::INFO),
                    service_info
                );
            });
        }
    }
}
