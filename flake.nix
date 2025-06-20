{
  description = "A devShell example";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    nixpkgs,
    rust-overlay,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        overlays = [(import rust-overlay)];
        pkgs = import nixpkgs {
          inherit system overlays;
        };
      in {
        devShells.default = with pkgs;
          mkShell.override {stdenv = clangStdenv;} {
            buildInputs = [
              # openssl
              # pkg-config
              # eza
              # fd
              # spirv-tools
              (rust-bin.nightly."2024-11-22".default.override {
                extensions = ["rust-src" "rustc-dev" "llvm-tools"];
              })
              mujoco
            ];

            LD_LIBRARY_PATH = with pkgs;
              lib.makeLibraryPath [
                libGL
                libxkbcommon
                wayland
              ];

            shellHook = ''
              alias ls=eza
              alias find=fd
              export MUJOCO_DIR=${pkgs.mujoco}
              export LIBCLANG_PATH=${pkgs.libclang.lib}/lib
            '';
          };
      }
    );
}
