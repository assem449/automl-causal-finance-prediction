#!/usr/bin/env python3
"""
Project Navigation Helper
Helps users navigate the organized project structure and run different components.
"""

import os
import sys
import subprocess
from pathlib import Path

class ProjectNavigator:
    """
    Helper class to navigate the organized project structure.
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.src_dir = self.project_root / "src"
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.outputs_dir = self.project_root / "outputs"
        self.docs_dir = self.project_root / "docs"
        self.notebooks_dir = self.project_root / "notebooks"
    
    def show_project_structure(self):
        """
        Display the current project structure.
        """
        print("📁 PROJECT STRUCTURE")
        print("=" * 50)
        
        print(f"\n🏠 Project Root: {self.project_root}")
        
        print(f"\n📊 Data Directory ({self.data_dir}):")
        if self.data_dir.exists():
            for file in self.data_dir.glob("*.csv"):
                print(f"  📄 {file.name}")
        else:
            print("  ⚠️  Data directory not found")
        
        print(f"\n🤖 Models Directory ({self.models_dir}):")
        if self.models_dir.exists():
            model_files = list(self.models_dir.glob("*.pth"))
            if model_files:
                for file in model_files:
                    print(f"  🧠 {file.name}")
            else:
                print("  📝 No trained models found")
        else:
            print("  ⚠️  Models directory not found")
        
        print(f"\n📈 Outputs Directory ({self.outputs_dir}):")
        if self.outputs_dir.exists():
            png_files = list(self.outputs_dir.glob("*.png"))
            json_files = list(self.outputs_dir.glob("*.json"))
            if png_files or json_files:
                print(f"  🖼️  {len(png_files)} visualization files")
                print(f"  📋 {len(json_files)} result files")
            else:
                print("  📝 No output files found")
        else:
            print("  ⚠️  Outputs directory not found")
        
        print(f"\n📚 Documentation ({self.docs_dir}):")
        if self.docs_dir.exists():
            for file in self.docs_dir.glob("*.md"):
                print(f"  📖 {file.name}")
        else:
            print("  ⚠️  Documentation directory not found")
        
        print(f"\n💻 Source Code ({self.src_dir}):")
        if self.src_dir.exists():
            py_files = list(self.src_dir.glob("*.py"))
            print(f"  🐍 {len(py_files)} Python files")
        else:
            print("  ⚠️  Source directory not found")
    
    def list_available_scripts(self):
        """
        List all available Python scripts in the src directory.
        """
        print("\n🔧 AVAILABLE SCRIPTS")
        print("=" * 50)
        
        if not self.src_dir.exists():
            print("❌ Source directory not found")
            return
        
        scripts = {
            "Data Management": [
                "fetch_fred_data.py",
                "preprocess_data.py",
                "simple_preprocess.py"
            ],
            "Model Development": [
                "lstm_timeseries.py",
                "train_lstm.py",
                "training_example.py",
                "lstm_example.py"
            ],
            "Hyperparameter Optimization": [
                "optuna_lstm_tuning.py",
                "optuna_example.py"
            ],
            "Causal Analysis": [
                "causal_dag.py",
                "simple_causal_dag.py"
            ],
            "Fairness Evaluation": [
                "add_fairness_groups.py",
                "create_fairness_dataset.py",
                "fairness_evaluation.py",
                "fairness_example.py",
                "fairlearn_evaluation.py",
                "fairlearn_example.py"
            ],
            "Interpretability": [
                "lstm_interpretability.py",
                "interpretability_example.py"
            ],
            "Utilities": [
                "setup_env.py",
                "example_usage.py",
                "project_navigation.py"
            ]
        }
        
        for category, files in scripts.items():
            print(f"\n📂 {category}:")
            for file in files:
                file_path = self.src_dir / file
                if file_path.exists():
                    print(f"  ✅ {file}")
                else:
                    print(f"  ❌ {file} (not found)")
    
    def run_script(self, script_name, *args):
        """
        Run a Python script from the src directory.
        
        Parameters:
        -----------
        script_name : str
            Name of the script to run
        *args : tuple
            Additional arguments to pass to the script
        """
        script_path = self.src_dir / script_name
        
        if not script_path.exists():
            print(f"❌ Script not found: {script_path}")
            return False
        
        print(f"🚀 Running {script_name}...")
        print(f"📁 Path: {script_path}")
        
        try:
            # Change to project root directory
            os.chdir(self.project_root)
            
            # Run the script
            cmd = [sys.executable, str(script_path)] + list(args)
            result = subprocess.run(cmd, capture_output=False)
            
            if result.returncode == 0:
                print(f"✅ {script_name} completed successfully")
                return True
            else:
                print(f"❌ {script_name} failed with return code {result.returncode}")
                return False
                
        except Exception as e:
            print(f"❌ Error running {script_name}: {e}")
            return False
    
    def show_quick_start_guide(self):
        """
        Display a quick start guide for the project.
        """
        print("\n🚀 QUICK START GUIDE")
        print("=" * 50)
        
        print("\n1️⃣ Environment Setup:")
        print("   python src/setup_env.py")
        print("   pip install -r requirements.txt")
        
        print("\n2️⃣ Data Fetching:")
        print("   python src/fetch_fred_data.py")
        
        print("\n3️⃣ Data Preprocessing:")
        print("   python src/preprocess_data.py")
        
        print("\n4️⃣ Model Training:")
        print("   python src/train_lstm.py")
        
        print("\n5️⃣ Hyperparameter Optimization:")
        print("   python src/optuna_lstm_tuning.py")
        
        print("\n6️⃣ Fairness Evaluation:")
        print("   python src/create_fairness_dataset.py")
        print("   python src/fairness_evaluation.py")
        
        print("\n7️⃣ Interpretability Analysis:")
        print("   python src/lstm_interpretability.py")
        
        print("\n📚 For detailed documentation, see docs/ directory")
    
    def check_data_files(self):
        """
        Check if required data files exist.
        """
        print("\n📊 DATA FILES CHECK")
        print("=" * 50)
        
        required_files = [
            "cleaned_macroeconomic_data.csv",
            "macroeconomic_data.csv",
            "macroeconomic_data_with_fairness_groups.csv"
        ]
        
        for file in required_files:
            file_path = self.data_dir / file
            if file_path.exists():
                size = file_path.stat().st_size
                print(f"✅ {file} ({size:,} bytes)")
            else:
                print(f"❌ {file} (missing)")
    
    def check_model_files(self):
        """
        Check if trained model files exist.
        """
        print("\n🤖 MODEL FILES CHECK")
        print("=" * 50)
        
        model_files = list(self.models_dir.glob("*.pth"))
        
        if model_files:
            for file in model_files:
                size = file.stat().st_size
                print(f"✅ {file.name} ({size:,} bytes)")
        else:
            print("📝 No trained models found")
            print("💡 Run training scripts to generate models")
    
    def check_output_files(self):
        """
        Check if output files exist.
        """
        print("\n📈 OUTPUT FILES CHECK")
        print("=" * 50)
        
        png_files = list(self.outputs_dir.glob("*.png"))
        json_files = list(self.outputs_dir.glob("*.json"))
        
        if png_files:
            print(f"🖼️  {len(png_files)} visualization files:")
            for file in png_files[:5]:  # Show first 5
                size = file.stat().st_size
                print(f"  📄 {file.name} ({size:,} bytes)")
            if len(png_files) > 5:
                print(f"  ... and {len(png_files) - 5} more")
        else:
            print("📝 No visualization files found")
        
        if json_files:
            print(f"\n📋 {len(json_files)} result files:")
            for file in json_files:
                size = file.stat().st_size
                print(f"  📄 {file.name} ({size:,} bytes)")
        else:
            print("📝 No result files found")

def main():
    """
    Main function to run the project navigator.
    """
    navigator = ProjectNavigator()
    
    print("🧭 PROJECT NAVIGATOR")
    print("=" * 60)
    
    # Show project structure
    navigator.show_project_structure()
    
    # List available scripts
    navigator.list_available_scripts()
    
    # Show quick start guide
    navigator.show_quick_start_guide()
    
    # Check files
    navigator.check_data_files()
    navigator.check_model_files()
    navigator.check_output_files()
    
    print("\n" + "=" * 60)
    print("✅ Project navigation complete!")
    print("💡 Use 'python src/project_navigation.py' to see this overview again")

if __name__ == "__main__":
    main() 