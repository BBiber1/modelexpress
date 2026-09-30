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
    let subscriber = tracing_subscriber::registry()
        .with(EnvFilter::from_default_env().add_directive(log_level.into()))
        .with(tracing_subscriber::fmt::layer())
        .with(otel_layer);
    tracing::subscriber::set_global_default(subscriber)?;

    // Translate a CTRL+C into a graceful shutdown of the embedded server.
    let shutdown = async {
        if let Err(e) = tokio::signal::ctrl_c().await {
            error!("Failed to install CTRL+C signal handler: {e}");
            return;
        }
        info!("Received CTRL+C, shutting down gracefully...");
    };

    let backend = BackendConfig::from_env()?;

    let result = run_server(config, backend, shutdown).await;
    if let Some(provider) = trace_provider {
        provider.shutdown()?;
    }
    result
}
