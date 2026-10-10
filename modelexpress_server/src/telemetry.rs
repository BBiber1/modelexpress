// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! W3C propagation for refit gRPC calls.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use opentelemetry::KeyValue;
use opentelemetry::baggage::BaggageExt;
use opentelemetry::propagation::TextMapPropagator;
use opentelemetry::trace::TraceContextExt;
use opentelemetry_sdk::propagation::{BaggagePropagator, TraceContextPropagator};
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
        let carrier: HashMap<String, String> = ["traceparent", "tracestate", "baggage"]
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
                role = "server",
                rank = 0,
                experiment = experiment,
                staging_mode = staging_mode,
                rpc.method = method,
                rpc.outcome = field::Empty
            );
            if carrier.contains_key("traceparent") {
                let parent = TraceContextPropagator::new().extract(&carrier);
                let parent = BaggagePropagator::new().extract_with_context(&parent, &carrier);
                let _ = span.set_parent(parent.clone());
                for key in [
                    "experiment",
                    "step",
                    "version_uid",
                    "staging_mode",
                    "refit.id",
                    "refit.step",
                    "refit.phase",
                    "mx.experiment.run_id",
                    "refit.aggregate",
                ] {
                    if let Some(value) = parent.baggage().get(key) {
                        if matches!(key, "step" | "refit.step") {
                            if let Ok(number) = value.as_str().parse::<i64>() {
                                span.context()
                                    .span()
                                    .set_attribute(KeyValue::new(key, number));
                            }
                        } else {
                            span.context()
                                .span()
                                .set_attribute(KeyValue::new(key, value.to_string()));
                        }
                    }
                }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry_sdk::error::OTelSdkResult;
    use opentelemetry_sdk::trace::{SdkTracerProvider, SpanData, SpanExporter};
    use tracing::instrument::WithSubscriber;
    use tracing_subscriber::layer::SubscriberExt;

    #[derive(Debug)]
    struct Exporter(mpsc::Sender<Vec<SpanData>>);

    impl SpanExporter for Exporter {
        #[allow(clippy::expect_used)]
        async fn export(&self, spans: Vec<SpanData>) -> OTelSdkResult {
            self.0.send(spans).expect("record exported spans");
            Ok(())
        }
    }

    #[tokio::test]
    #[allow(clippy::expect_used)]
    async fn refit_layer_exports_remote_parent_and_grpc_outcome_only_when_enabled() {
        for (enabled, path, exported) in [
            (
                true,
                "/model_express.refit.RefitService/GetWeightVersion",
                true,
            ),
            (
                false,
                "/model_express.refit.RefitService/GetWeightVersion",
                false,
            ),
            (true, "/model_express.model.ModelService/GetModel", false),
        ] {
            let (sender, receiver) = mpsc::channel();
            let provider = SdkTracerProvider::builder()
                .with_simple_exporter(Exporter(sender))
                .build();
            let subscriber = tracing_subscriber::registry()
                .with(tracing_opentelemetry::layer().with_tracer(provider.tracer("test")));
            let inner = tower::service_fn(|_request: http::Request<()>| async {
                let mut response = http::Response::new(());
                response
                    .extensions_mut()
                    .insert(tonic::Status::failed_precondition("retiring"));
                Ok::<_, std::convert::Infallible>(response)
            });
            let mut service = RefitTraceLayer::new(enabled).layer(inner);
            let request = http::Request::builder()
                .uri(path)
                .header(
                    "traceparent",
                    "00-11111111111111111111111111111111-2222222222222222-01",
                )
                .header("baggage", "step=7,experiment=refit-test,refit.id=version-a")
                .body(())
                .expect("valid request");
            let response = service
                .call(request)
                .with_subscriber(subscriber)
                .await
                .expect("service response");
            assert_eq!(
                response
                    .extensions()
                    .get::<tonic::Status>()
                    .expect("response status")
                    .code(),
                tonic::Code::FailedPrecondition
            );
            provider.force_flush().expect("flush spans");
            let spans: Vec<_> = receiver.try_iter().flatten().collect();
            assert_eq!(spans.len(), usize::from(exported));
            if let Some(span) = spans.first() {
                assert_eq!(span.name, "mx.refit.grpc");
                assert_eq!(
                    span.span_context.trace_id().to_string(),
                    "11111111111111111111111111111111"
                );
                assert_eq!(span.parent_span_id.to_string(), "2222222222222222");
                assert!(span.parent_span_is_remote);
                assert!(span.attributes.contains(&KeyValue::new("step", 7_i64)));
                assert!(
                    span.attributes
                        .contains(&KeyValue::new("refit.id", "version-a"))
                );
                assert!(span.attributes.contains(&KeyValue::new("role", "server")));
                assert!(span.attributes.contains(&KeyValue::new(
                    "rpc.outcome",
                    tonic::Code::FailedPrecondition.to_string()
                )));
            }
            provider.shutdown().expect("shutdown provider");
        }
    }
}
