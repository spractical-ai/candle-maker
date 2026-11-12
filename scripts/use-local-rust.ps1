# Dot-source this script: . .\scripts\use-local-rust.ps1
$candleProjectRoot = Split-Path -Parent $PSScriptRoot
$env:CARGO_HOME = Join-Path $candleProjectRoot '.tools\cargo'
$env:RUSTUP_HOME = Join-Path $candleProjectRoot '.tools\rustup'
$env:PATH = (Join-Path $env:CARGO_HOME 'bin') + [IO.Path]::PathSeparator + $env:PATH

# Cargo does not automatically use Windows' system proxy settings.
if (-not $env:CARGO_HTTP_PROXY) {
    $candleRegistryUri = [Uri]'https://index.crates.io/config.json'
    $candleRegistryProxy = [System.Net.WebRequest]::DefaultWebProxy.GetProxy($candleRegistryUri)
    if ($candleRegistryProxy.Host -ne $candleRegistryUri.Host) {
        $env:CARGO_HTTP_PROXY = $candleRegistryProxy.AbsoluteUri
    }
}
