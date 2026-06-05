fn main() {
    let config = slint_build::CompilerConfiguration::new().with_style("fluent-dark".to_string());
    slint_build::compile_with_config("ui/main.slint", config).unwrap();

    if let Ok(output) = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
    {
        let hash = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !hash.is_empty() {
            println!("cargo:rustc-env=GIT_HASH={hash}");
        }
    }
    println!("cargo:rerun-if-changed=../../.git/HEAD");

    // Increment and read build number
    let build_number_file = std::path::Path::new(".build_number");
    let mut build_number: u32 = 3; // Start at 03

    if let Ok(content) = std::fs::read_to_string(build_number_file) {
        if let Ok(num) = content.trim().parse::<u32>() {
            build_number = num + 1;
        }
    }

    let _ = std::fs::write(build_number_file, format!("{}", build_number));
    println!("cargo:rustc-env=BUILD_NUMBER={:02}", build_number);
}
