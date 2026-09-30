// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! W3C propagation for refit gRPC calls.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use opentelemetry::propagation::TextMapPropagator;
use opentelemetry::trace::TraceContextExt;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use tower::{Layer, Service};
use tracing::{Instrument, field};
use tracing_opentelemetry::OpenTelemetrySpanExt;

#[derive(Clone)]
pub struct RefitTraceLayer {
    enabled: bool,
}

impl RefitTraceLayer {
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }
}

impl<S> Layer<S> for RefitTraceLayer {
    type Service = RefitTraceService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        RefitTraceService {
            inner,
            enabled: self.enabled,
        }
    }
}

#[derive(Clone)]
pub struct RefitTraceService<S> {
    inner: S,
    enabled: bool,
}

impl<S, ReqBody, ResBody> Service<http::Request<ReqBody>> for RefitTraceService<S>
where
    S: Service<http::Request<ReqBody>, Response = http::Response<ResBody>> + Clone + Send + 'static,
    S::Future: Send + 'static,
    ReqBody: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, request: http::Request<ReqBody>) -> Self::Future {
        let is_refit = request
            .uri()
            .path()
            .starts_with("/model_express.refit.RefitService/");
        let method = crate::metrics::grpc::method_label(request.uri().path());
        let carrier: HashMap<String, String> = ["traceparent", "tracestate"]
            .into_iter()
            .filter_map(|key| {
                request
                    .headers()
                    .get(key)
                    .and_then(|value| value.to_str().ok())
                    .map(|value| (key.to_string(), value.to_string()))
            })
            .collect();

        let ready = self.inner.clone();
        let mut inner = std::mem::replace(&mut self.inner, ready);
        if !self.enabled || !is_refit {
            return Box::pin(async move { inner.call(request).await });
        }

        Box::pin(async move {
            let experiment = std::env::var("MX_REFIT_EXPERIMENT").unwrap_or_default();
            let staging_mode = std::env::var("MX_REFIT_STAGING_MODE").unwrap_or_default();
            let span = tracing::info_span!(
                "mx.refit.grpc",
                role = "control",
                rank = 0,
                experiment = experiment,
                staging_mode = staging_mode,
                rpc.method = method,
                rpc.outcome = field::Empty
            );
            if carrier.contains_key("traceparent") {
                let parent = TraceContextPropagator::new().extract(&carrier);
                let _ = span.set_parent(parent);
            }
            let result = inner.call(request).instrument(span.clone()).await;
            if span.context().span().is_recording() {
                let outcome = match &result {
                    Ok(response) => response
                        .extensions()
                        .get::<tonic::Status>()
                        .map_or_else(|| "ok".to_string(), |status| status.code().to_string()),
                    Err(_) => "error".to_string(),
                };
                span.record("rpc.outcome", outcome.as_str());
            }
            result
        })
    }
}
