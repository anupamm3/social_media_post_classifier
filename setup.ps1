# Mental Health Tweet Classifier - PowerShell Setup Script
# Windows-compatible alternative to Makefile

param(
    [string]$Command = "help",
    [string]$Model = "baseline",
    [string]$Data = "dataset",
    [string]$Output = "models"
)

# Colors for output
$Red = "`e[91m"
$Green = "`e[92m" 
$Yellow = "`e[93m"
$Blue = "`e[94m"
$Magenta = "`e[95m"
$Cyan = "`e[96m"
$White = "`e[97m"
$Reset = "`e[0m"

# Project configuration
$ProjectName = "mental_health_tweet_classifier"
$SrcDir = "src"
$DataDir = "dataset"
$ModelsDir = "models"
$ReportsDir = "reports"
$NotebooksDir = "notebooks"
$TestsDir = "tests"

function Write-ColorOutput {
    param([string]$Message, [string]$Color = $White)
    Write-Host "$Color$Message$Reset"
}

function Write-Header {
    param([string]$Title)
    Write-ColorOutput "`n$('=' * 60)" $Cyan
    Write-ColorOutput "$Title" $Cyan
    Write-ColorOutput "$('=' * 60)" $Cyan
}

function Write-Success {
    param([string]$Message)
    Write-ColorOutput "✅ $Message" $Green
}

function Write-Error {
    param([string]$Message)
    Write-ColorOutput "❌ $Message" $Red
}

function Write-Warning {
    param([string]$Message)
    Write-ColorOutput "⚠️  $Message" $Yellow
}

function Write-Info {
    param([string]$Message)
    Write-ColorOutput "ℹ️  $Message" $Blue
}

function Test-PythonPackage {
    param([string]$Package)
    try {
        $result = python -c "import $Package; print('OK')" 2>$null
        return $result -eq "OK"
    }
    catch {
        return $false
    }
}

function Setup-Project {
    Write-Header "Setting up $ProjectName"
    
    # Check Python installation
    Write-Info "Checking Python installation..."
    try {
        $pythonVersion = python --version 2>&1
        Write-Success "Python found: $pythonVersion"
    }
    catch {
        Write-Error "Python not found. Please install Python 3.8+ and ensure it's in your PATH."
        return
    }
    
    # Upgrade pip
    Write-Info "Upgrading pip..."
    python -m pip install --upgrade pip
    
    # Install requirements
    if (Test-Path "requirements.txt") {
        Write-Info "Installing Python packages from requirements.txt..."
        python -m pip install -r requirements.txt
        Write-Success "Packages installed successfully"
    }
    else {
        Write-Warning "requirements.txt not found, installing basic packages..."
        python -m pip install pandas numpy scikit-learn matplotlib seaborn
    }
    
    # Download NLTK data
    Write-Info "Downloading NLTK data..."
    try {
        python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"
        Write-Success "NLTK data downloaded"
    }
    catch {
        Write-Warning "Could not download NLTK data. Some features may be limited."
    }
    
    # Create directories
    Write-Info "Creating project directories..."
    $dirs = @($ModelsDir, $ReportsDir, "logs", "experiments", "explanations", "evaluation_results")
    foreach ($dir in $dirs) {
        if (!(Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Info "Created directory: $dir"
        }
    }
    
    Write-Success "Setup complete! 🎉"
    Write-Info "Next steps:"
    Write-Info "  1. Run: .\setup.ps1 explore-data"
    Write-Info "  2. Run: .\setup.ps1 train-baseline"
    Write-Info "  3. Run: .\setup.ps1 demo"
}

function Setup-Dev {
    Write-Header "Setting up development environment"
    Setup-Project
    
    Write-Info "Installing development packages..."
    python -m pip install pytest black flake8 mypy jupyter
    Write-Success "Development environment ready!"
}

function Explore-Data {
    Write-Header "Running Exploratory Data Analysis"
    
    if (Test-Path "$NotebooksDir\01_explore.ipynb") {
        Write-Info "Executing EDA notebook..."
        try {
            jupyter nbconvert --to notebook --execute "$NotebooksDir\01_explore.ipynb" --output "01_explore_executed.ipynb"
            Write-Success "EDA complete! Check $NotebooksDir\01_explore_executed.ipynb"
        }
        catch {
            Write-Error "Failed to execute notebook. Make sure Jupyter is installed."
            Write-Info "Alternative: Open $NotebooksDir\01_explore.ipynb manually in Jupyter"
        }
    }
    else {
        Write-Error "EDA notebook not found at $NotebooksDir\01_explore.ipynb"
    }
}

function Train-Baseline {
    Write-Header "Training Baseline Models"
    
    Write-Info "Training TF-IDF + Logistic Regression/Random Forest..."
    try {
        python -m $SrcDir.models.baseline --data $DataDir --output "$ModelsDir\baseline"
        Write-Success "Baseline models trained successfully!"
    }
    catch {
        Write-Error "Baseline training failed. Check error messages above."
        Write-Info "Make sure all required packages are installed with: .\setup.ps1 setup"
    }
}

function Train-Transformer {
    Write-Header "Training Transformer Model"
    
    Write-Info "Training BERT/RoBERTa model..."
    Write-Warning "This may take 30-120 minutes depending on your hardware"
    
    try {
        python -m $SrcDir.models.transformer --data $DataDir --output "$ModelsDir\transformer"
        Write-Success "Transformer model trained successfully!"
    }
    catch {
        Write-Error "Transformer training failed. Check error messages above."
        Write-Info "Make sure transformers and torch are installed"
    }
}

function Train-All {
    Train-Baseline
    Train-Transformer
}

function Evaluate-Model {
    param([string]$ModelType = $Model)
    
    Write-Header "Evaluating $ModelType Model"
    
    if (!(Test-Path "$ModelsDir\$ModelType")) {
        Write-Error "Model not found at $ModelsDir\$ModelType"
        Write-Info "Train the model first with: .\setup.ps1 train-$ModelType"
        return
    }
    
    try {
        python -m $SrcDir.eval.evaluate --model "$ModelsDir\$ModelType" --data $DataDir --output "$ReportsDir\${ModelType}_evaluation"
        Write-Success "$ModelType evaluation complete!"
    }
    catch {
        Write-Error "Evaluation failed. Check error messages above."
    }
}

function Evaluate-All {
    Evaluate-Model -ModelType "baseline"
    Evaluate-Model -ModelType "transformer"
}

function Explain-Predictions {
    Write-Header "Generating Model Explanations"
    
    try {
        python -m $SrcDir.eval.explain --model $ModelsDir --data $DataDir --output "$ReportsDir\explanations"
        Write-Success "Explanations generated successfully!"
    }
    catch {
        Write-Error "Explanation generation failed. Check error messages above."
    }
}

function Start-Demo {
    Write-Header "Starting Streamlit Demo"
    
    if (!(Test-Path "app\streamlit_app.py")) {
        Write-Error "Streamlit app not found at app\streamlit_app.py"
        return
    }
    
    # Check if streamlit is installed
    if (!(Test-PythonPackage "streamlit")) {
        Write-Warning "Streamlit not installed. Installing now..."
        python -m pip install streamlit
    }
    
    Write-Info "Starting Streamlit demo at http://localhost:8501"
    Write-Warning "⚠️  REMEMBER: This is for research/educational purposes ONLY!"
    Write-Warning "   NOT for clinical diagnosis or medical decisions."
    
    try {
        streamlit run app\streamlit_app.py --server.port 8501
    }
    catch {
        Write-Error "Failed to start Streamlit. Make sure it's installed and try again."
    }
}

function Start-Notebook {
    Write-Header "Starting Jupyter Notebook"
    
    if (!(Test-PythonPackage "jupyter")) {
        Write-Warning "Jupyter not installed. Installing now..."
        python -m pip install jupyter
    }
    
    try {
        jupyter notebook --notebook-dir=$NotebooksDir
    }
    catch {
        Write-Error "Failed to start Jupyter. Make sure it's installed."
    }
}

function Run-Tests {
    Write-Header "Running Unit Tests"
    
    if (!(Test-Path $TestsDir)) {
        Write-Warning "Tests directory not found. Creating basic test structure..."
        New-Item -ItemType Directory -Path $TestsDir -Force | Out-Null
        # Create basic test file
        @"
import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

def test_basic_imports():
    """Test that basic modules can be imported."""
    try:
        from src.data import load_data
        from src.data import preprocess
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_project_structure():
    """Test that expected directories exist."""
    project_root = Path(__file__).parent.parent
    expected_dirs = ["src", "dataset", "notebooks", "app"]
    
    for dir_name in expected_dirs:
        dir_path = project_root / dir_name
        assert dir_path.exists(), f"Expected directory {dir_name} not found"

if __name__ == "__main__":
    test_basic_imports()
    test_project_structure()
    print("✅ Basic tests passed!")
"@ | Out-File -FilePath "$TestsDir\test_basic.py" -Encoding UTF8
    }
    
    if (!(Test-PythonPackage "pytest")) {
        Write-Warning "pytest not installed. Installing now..."
        python -m pip install pytest
    }
    
    try {
        python -m pytest $TestsDir -v
        Write-Success "Tests completed!"
    }
    catch {
        Write-Error "Tests failed. Check error messages above."
    }
}

function Format-Code {
    Write-Header "Formatting Code"
    
    if (!(Test-PythonPackage "black")) {
        Write-Warning "black not installed. Installing now..."
        python -m pip install black
    }
    
    try {
        python -m black $SrcDir app\ --line-length 100
        Write-Success "Code formatting complete!"
    }
    catch {
        Write-Error "Code formatting failed."
    }
}

function Clean-Project {
    Write-Header "Cleaning Project"
    
    $cleanDirs = @("__pycache__", "*.pyc", ".pytest_cache", ".mypy_cache", "build", "dist", "*.egg-info")
    $cleanPaths = @("$ModelsDir\*", "$ReportsDir\*", "logs\*", "experiments\*")
    
    Write-Info "Removing Python cache files..."
    Get-ChildItem -Path . -Recurse -Force | Where-Object { $_.Name -eq "__pycache__" -or $_.Extension -eq ".pyc" } | Remove-Item -Recurse -Force
    
    Write-Info "Cleaning generated files..."
    foreach ($path in $cleanPaths) {
        if (Test-Path $path) {
            Remove-Item $path -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
    
    Write-Success "Cleanup complete!"
}

function Show-Help {
    Write-Header "Mental Health Tweet Classifier - PowerShell Commands"
    
    Write-ColorOutput "`n📊 SETUP COMMANDS:" $Magenta
    Write-Info "  .\setup.ps1 setup          - Install dependencies and set up project"
    Write-Info "  .\setup.ps1 setup-dev      - Install development dependencies"
    
    Write-ColorOutput "`n📈 DATA COMMANDS:" $Magenta  
    Write-Info "  .\setup.ps1 explore-data   - Run exploratory data analysis"
    
    Write-ColorOutput "`n🤖 TRAINING COMMANDS:" $Magenta
    Write-Info "  .\setup.ps1 train-baseline - Train baseline models (TF-IDF + ML)"
    Write-Info "  .\setup.ps1 train-transformer - Train transformer model (BERT/RoBERTa)"
    Write-Info "  .\setup.ps1 train-all      - Train all models"
    
    Write-ColorOutput "`n📊 EVALUATION COMMANDS:" $Magenta
    Write-Info "  .\setup.ps1 evaluate -Model baseline    - Evaluate baseline models"
    Write-Info "  .\setup.ps1 evaluate -Model transformer - Evaluate transformer model"  
    Write-Info "  .\setup.ps1 evaluate-all   - Evaluate all models"
    Write-Info "  .\setup.ps1 explain        - Generate model explanations (SHAP/LIME)"
    
    Write-ColorOutput "`n🖥️ DEMO COMMANDS:" $Magenta
    Write-Info "  .\setup.ps1 demo           - Start Streamlit demo app"
    Write-Info "  .\setup.ps1 notebook       - Start Jupyter notebook server"
    
    Write-ColorOutput "`n🧪 TESTING COMMANDS:" $Magenta
    Write-Info "  .\setup.ps1 test           - Run unit tests"
    Write-Info "  .\setup.ps1 format         - Format code with Black"
    
    Write-ColorOutput "`n🧹 UTILITY COMMANDS:" $Magenta
    Write-Info "  .\setup.ps1 clean          - Clean up generated files"
    Write-Info "  .\setup.ps1 info           - Show project information"
    Write-Info "  .\setup.ps1 help           - Show this help message"
    
    Write-ColorOutput "`n🚀 QUICK START:" $Cyan
    Write-Info "  1. .\setup.ps1 setup       # Install dependencies"
    Write-Info "  2. .\setup.ps1 train-baseline  # Train model"  
    Write-Info "  3. .\setup.ps1 demo        # Start demo app"
    
    Write-ColorOutput "`n⚠️ IMPORTANT REMINDER:" $Red
    Write-Info "  This tool is for RESEARCH/EDUCATIONAL purposes ONLY!"
    Write-Info "  NOT for clinical diagnosis or medical decisions."
    Write-Info "  See docs\ethics.md for full guidelines."
}

function Show-Info {
    Write-Header "Project Information"
    
    $pythonVersion = try { python --version 2>&1 } catch { "Not found" }
    $pipVersion = try { python -m pip --version 2>&1 } catch { "Not found" }
    
    Write-ColorOutput "`n📊 Mental Health Tweet Classifier Project" $Cyan
    Write-ColorOutput "=========================================" $Cyan
    Write-Info "Project: $ProjectName"
    Write-Info "Python: $pythonVersion"
    Write-Info "Pip: $pipVersion"
    
    Write-ColorOutput "`n📁 Project Structure:" $Magenta
    Write-Info "  $SrcDir\          - Source code"
    Write-Info "  $DataDir\         - Dataset files"  
    Write-Info "  $ModelsDir\       - Trained models"
    Write-Info "  $ReportsDir\      - Evaluation reports"
    Write-Info "  $NotebooksDir\    - Jupyter notebooks"
    Write-Info "  $TestsDir\        - Unit tests"
    Write-Info "  app\             - Demo applications"
    Write-Info "  docs\            - Documentation"
    
    Write-ColorOutput "`n📊 Dataset Status:" $Magenta
    if (Test-Path $DataDir) {
        $csvFiles = Get-ChildItem "$DataDir\*.csv" | Measure-Object
        Write-Info "  CSV files found: $($csvFiles.Count)"
        
        Get-ChildItem "$DataDir\*.csv" | ForEach-Object {
            $lineCount = (Get-Content $_.FullName | Measure-Object).Count - 1  # Subtract header
            Write-Info "    $($_.Name): ~$lineCount rows"
        }
    }
    else {
        Write-Warning "  Dataset directory not found"
    }
    
    Write-ColorOutput "`n🤖 Model Status:" $Magenta
    if (Test-Path $ModelsDir) {
        $modelDirs = Get-ChildItem $ModelsDir -Directory 2>$null
        if ($modelDirs) {
            foreach ($dir in $modelDirs) {
                Write-Info "  ✅ $($dir.Name) model available"
            }
        }
        else {
            Write-Warning "  No trained models found"
            Write-Info "  Run: .\setup.ps1 train-baseline to get started"
        }
    }
    else {
        Write-Warning "  Models directory not found"
    }
    
    Write-ColorOutput "`n⚠️ IMPORTANT:" $Red
    Write-Info "  This is for research and educational purposes ONLY."
    Write-Info "  NOT for clinical diagnosis or medical advice."
    Write-Info "  See docs\ethics.md for complete guidelines."
}

function Run-Pipeline {
    Write-Header "Running Full ML Pipeline"
    Setup-Project
    Explore-Data  
    Train-All
    Evaluate-All
    Explain-Predictions
    Write-Success "Full ML pipeline complete! 🎉"
}

function Quick-Start {
    Write-Header "Quick Start Demo"
    Setup-Project
    Train-Baseline
    Evaluate-Model -ModelType "baseline"
    Start-Demo
}

# Main command dispatcher
switch ($Command.ToLower()) {
    "setup" { Setup-Project }
    "setup-dev" { Setup-Dev }
    "explore-data" { Explore-Data }
    "train-baseline" { Train-Baseline }
    "train-transformer" { Train-Transformer }
    "train-all" { Train-All }
    "evaluate" { Evaluate-Model -ModelType $Model }
    "evaluate-all" { Evaluate-All }
    "explain" { Explain-Predictions }
    "demo" { Start-Demo }
    "notebook" { Start-Notebook }
    "test" { Run-Tests }
    "format" { Format-Code }
    "clean" { Clean-Project }
    "info" { Show-Info }
    "help" { Show-Help }
    "pipeline" { Run-Pipeline }
    "quick-start" { Quick-Start }
    default { 
        Write-Error "Unknown command: $Command"
        Show-Help
    }
}