{
  description = "simple rl model eval environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux"; # Adjust to "aarch64-darwin", etc., as required
      pkgs = nixpkgs.legacyPackages.${system};
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = [
          (pkgs.python3.withPackages (ps: with ps; [
            numpy
            matplotlib
            pandas
            ipython
          ]))
        ];

        shellHook = ''
          echo "environment loaded."
          python --version
        '';
      };
    };
}
