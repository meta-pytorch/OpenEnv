// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

fn main() -> std::io::Result<()> {
    let proto_file = "agent_bus.proto";
    println!("cargo:rerun-if-changed={}", proto_file);

    setup_protoc_env();

    let tonic = tonic_build::configure();
    tonic.compile(&[proto_file], &["."] /* includes */)
}

/// Setup process level env vars for tonic to find protoc etc
fn setup_protoc_env() {
    let protoc_bin = protoc_bin_vendored::protoc_bin_path().unwrap();
    unsafe {
        std::env::set_var("PROTOC", protoc_bin);
    }

    let protoc_inc = protoc_bin_vendored::include_path().unwrap();
    let protoc_inc = protoc_inc.canonicalize().unwrap(); // protoc wants canonicalized paths
    unsafe {
        std::env::set_var("PROTOC_INCLUDE", protoc_inc);
    }
}
