// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/core/error.hpp"

#include <cstdio>

namespace sapient::core {

std::string debug_dims(const std::vector<size_t>& dims) {
    std::string s = "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i) s += ", ";
        s += std::to_string(dims[i]);
    }
    return s + "]";
}

namespace {
// Rust `{:?}` on a String: double-quoted with \" \\ \n \t \r \0 escaped; every other ASCII
// control byte (< 0x20 or == 0x7F) is escaped as `\u{XX}` (lowercase hex, no zero padding) like
// `char::escape_debug`. Non-ASCII scalars pass through as UTF-8 (Rust would `\u{…}`-escape
// non-printables outside ASCII too — IR-path only, not hit by any current caller).
std::string debug_str(const std::string& s) {
    std::string out = "\"";
    for (const char c : s) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\t':
            out += "\\t";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\0':
            out += "\\0";
            break;
        default:
            if (static_cast<unsigned char>(c) < 0x20 || c == 0x7F) {
                char buf[16];
                std::snprintf(buf, sizeof(buf), "\\u{%x}", static_cast<unsigned char>(c));
                out += buf;
            } else {
                out += c;
            }
        }
    }
    return out + "\"";
}
} // namespace

std::string Error::to_string() const {
    switch (code) {
    case ErrorCode::ShapeMismatch:
        return "Shape mismatch: expected " + debug_dims(expected_dims) + ", got " +
               debug_dims(got_dims);
    case ErrorCode::RankMismatch:
        return "Rank mismatch: expected " + std::to_string(expected_n) + ", got " +
               std::to_string(got_n);
    case ErrorCode::TypeMismatch:
        return "Type mismatch: expected " + a + ", got " + b;
    case ErrorCode::BroadcastError:
        return "Incompatible shapes for broadcasting: " + debug_dims(expected_dims) + " and " +
               debug_dims(got_dims);
    case ErrorCode::CyclicGraph:
        return "Graph contains a cycle \xE2\x80\x94 execution is impossible";
    case ErrorCode::NodeNotFound:
        return "Node " + debug_str(a) + " not found in graph";
    case ErrorCode::InvalidGraph:
        return "Graph validation failed: " + a;
    case ErrorCode::ShapeInferenceFailed:
        return "Shape inference failed for op '" + a + "': " + b;
    case ErrorCode::UnsupportedOp:
        return "Backend '" + a + "' does not support op '" + b + "'";
    case ErrorCode::BackendError:
        return "Backend error from '" + a + "': " + b;
    case ErrorCode::NoBackendAvailable:
        return "No suitable backend found for execution";
    case ErrorCode::AllocationFailed:
        return "Allocation failed: requested " + std::to_string(expected_n) + " bytes (alignment " +
               std::to_string(got_n) + ")";
    case ErrorCode::BufferSizeMismatch:
        return "Buffer size mismatch: expected " + std::to_string(expected_n) + " bytes, got " +
               std::to_string(got_n);
    case ErrorCode::PoolExhausted:
        return "Memory pool exhausted \xE2\x80\x94 consider increasing pool capacity";
    case ErrorCode::OnnxParseError:
        return "ONNX parse error: " + a;
    case ErrorCode::GgufParseError:
        return "GGUF parse error: " + a;
    case ErrorCode::SafetensorsParseError:
        return "Safetensors parse error: " + a;
    case ErrorCode::UnsupportedFormat:
        return "Unsupported model format: " + a;
    case ErrorCode::ModelNotFound:
        return "Model not found at path '" + a + "'";
    case ErrorCode::Io:
        return "IO error: " + a;
    case ErrorCode::DeadlineExceeded:
        return "Request timed out (deadline exceeded)";
    case ErrorCode::SchedulerShutdown:
        return "Batch scheduler is shut down";
    case ErrorCode::UninitializedRuntime:
        return "Runtime is not initialized \xE2\x80\x94 call Session::new() first";
    case ErrorCode::TelemetryError:
        return "Telemetry export failed: " + a;
    case ErrorCode::Internal:
        return "Internal error: " + a;
    }
    return "Internal error: <unknown error code>";
}

namespace {
Error with_code(ErrorCode c) {
    Error e;
    e.code = c;
    return e;
}
Error with_str(ErrorCode c, std::string s) {
    Error e = with_code(c);
    e.a = std::move(s);
    return e;
}
Error with_two(ErrorCode c, std::string s, std::string t) {
    Error e = with_str(c, std::move(s));
    e.b = std::move(t);
    return e;
}
Error with_nums(ErrorCode c, size_t x, size_t y) {
    Error e = with_code(c);
    e.expected_n = x;
    e.got_n = y;
    return e;
}
Error with_dims(ErrorCode c, std::vector<size_t> x, std::vector<size_t> y) {
    Error e = with_code(c);
    e.expected_dims = std::move(x);
    e.got_dims = std::move(y);
    return e;
}
} // namespace

Error Error::shape_mismatch(std::vector<size_t> expected, std::vector<size_t> got) {
    return with_dims(ErrorCode::ShapeMismatch, std::move(expected), std::move(got));
}
Error Error::rank_mismatch(size_t expected, size_t got) {
    return with_nums(ErrorCode::RankMismatch, expected, got);
}
Error Error::type_mismatch(std::string expected, std::string got) {
    return with_two(ErrorCode::TypeMismatch, std::move(expected), std::move(got));
}
Error Error::broadcast(std::vector<size_t> lhs, std::vector<size_t> rhs) {
    return with_dims(ErrorCode::BroadcastError, std::move(lhs), std::move(rhs));
}
Error Error::cyclic_graph() {
    return with_code(ErrorCode::CyclicGraph);
}
Error Error::node_not_found(std::string node) {
    return with_str(ErrorCode::NodeNotFound, std::move(node));
}
Error Error::invalid_graph(std::string msg) {
    return with_str(ErrorCode::InvalidGraph, std::move(msg));
}
Error Error::shape_inference_failed(std::string op, std::string reason) {
    return with_two(ErrorCode::ShapeInferenceFailed, std::move(op), std::move(reason));
}
Error Error::unsupported_op(std::string backend_name, std::string op) {
    return with_two(ErrorCode::UnsupportedOp, std::move(backend_name), std::move(op));
}
Error Error::backend(std::string backend_name, std::string message) {
    return with_two(ErrorCode::BackendError, std::move(backend_name), std::move(message));
}
Error Error::no_backend_available() {
    return with_code(ErrorCode::NoBackendAvailable);
}
Error Error::allocation_failed(size_t bytes, size_t align) {
    return with_nums(ErrorCode::AllocationFailed, bytes, align);
}
Error Error::buffer_size_mismatch(size_t expected, size_t got) {
    return with_nums(ErrorCode::BufferSizeMismatch, expected, got);
}
Error Error::pool_exhausted() {
    return with_code(ErrorCode::PoolExhausted);
}
Error Error::onnx_parse(std::string msg) {
    return with_str(ErrorCode::OnnxParseError, std::move(msg));
}
Error Error::gguf_parse(std::string msg) {
    return with_str(ErrorCode::GgufParseError, std::move(msg));
}
Error Error::safetensors_parse(std::string msg) {
    return with_str(ErrorCode::SafetensorsParseError, std::move(msg));
}
Error Error::unsupported_format(std::string fmt) {
    return with_str(ErrorCode::UnsupportedFormat, std::move(fmt));
}
Error Error::model_not_found(std::string path) {
    return with_str(ErrorCode::ModelNotFound, std::move(path));
}
Error Error::io(std::error_code ec, std::string message) {
    Error e = with_str(ErrorCode::Io, std::move(message));
    e.io_code = ec;
    return e;
}
Error Error::deadline_exceeded() {
    return with_code(ErrorCode::DeadlineExceeded);
}
Error Error::scheduler_shutdown() {
    return with_code(ErrorCode::SchedulerShutdown);
}
Error Error::uninitialized_runtime() {
    return with_code(ErrorCode::UninitializedRuntime);
}
Error Error::telemetry(std::string msg) {
    return with_str(ErrorCode::TelemetryError, std::move(msg));
}
Error Error::internal(std::string msg) {
    return with_str(ErrorCode::Internal, std::move(msg));
}

} // namespace sapient::core
