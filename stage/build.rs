fn main() {
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    assert_eq!(std::env::var("CARGO_CFG_TARGET_OS").unwrap(), "macos");
    // No third-party build or installer runs here: the archive is the repository's patched libtinynv
    // (driver/, built by install.sh), or QWEN_TINYNV_ARCHIVE.
    let archive = std::env::var("QWEN_TINYNV_ARCHIVE").unwrap_or_else(|_| {
        format!("{}/../driver/build/libtinynv.a", env!("CARGO_MANIFEST_DIR"))
    });
    println!("cargo:rerun-if-env-changed=QWEN_TINYNV_ARCHIVE");
    println!("cargo:rerun-if-changed={archive}");
    println!(
        "cargo:rustc-link-search=native={}",
        std::path::Path::new(&archive).parent().unwrap().display()
    );
    println!("cargo:rustc-link-lib=static=tinynv");
    for framework in ["Foundation", "IOKit", "CoreFoundation"] {
        println!("cargo:rustc-link-lib=framework={framework}");
    }
}
