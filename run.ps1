param (
    [string[]]$ArgsToPass = @()
)

# Docker image name
$imageName = "cdmo-project"

# Build the image
Write-Host "Building Docker image '$imageName'..."
docker build -t $imageName .

# Normalize path for Docker
$projectPath = $PWD.Path -replace '\\', '/'

# Define volume mounts
$codeMount = "${projectPath}:/app"
$licenseMount = "${projectPath}/gurobi.lic:/root/gurobi.lic"

# Build final argument list (starts with 'python')
$command = @("python") + $ArgsToPass

# Run container
Write-Host "Running Docker container with arguments: $command"
& docker run --rm -v $codeMount -v $licenseMount $imageName @command
