#!/usr/bin/env python3
"""
Simple code analysis script for the directories
"""

import os
import glob
import ast
import json
from typing import Dict, List, Any
from collections import defaultdict


def analyze_python_file(file_path: str) -> Dict[str, Any]:
    """Analyze a single Python file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Basic metrics
        lines = content.split('\n')
        line_count = len(lines)
        empty_lines = sum(1 for line in lines if not line.strip())
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        
        # Parse AST for more advanced metrics
        try:
            tree = ast.parse(content)
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            
            # Check for issues
            issues = []
            
            # Check for long functions
            for func in functions:
                func_lines = func.end_lineno - func.lineno if hasattr(func, 'end_lineno') else 0
                if func_lines > 50:
                    issues.append(f"Long function '{func.name}' ({func_lines} lines)")
            
            # Check for missing docstrings
            for func in functions:
                if not ast.get_docstring(func):
                    issues.append(f"Missing docstring in function '{func.name}'")
            
            for cls in classes:
                if not ast.get_docstring(cls):
                    issues.append(f"Missing docstring in class '{cls.name}'")
            
            # Calculate complexity (simplified)
            complexity = len(functions) + len(classes)
            
            return {
                "file": file_path,
                "lines": line_count,
                "empty_lines": empty_lines,
                "comment_lines": comment_lines,
                "functions": len(functions),
                "classes": len(classes),
                "issues": issues,
                "complexity": complexity,
                "quality_score": max(0, 100 - len(issues) * 5)
            }
        except SyntaxError as e:
            return {
                "file": file_path,
                "lines": line_count,
                "error": f"Syntax error: {str(e)}",
                "quality_score": 0
            }
    except Exception as e:
        return {
            "file": file_path,
            "error": f"Error reading file: {str(e)}",
            "quality_score": 0
        }


def analyze_directory(directory: str) -> Dict[str, Any]:
    """Analyze all Python files in a directory"""
    results = []
    total_issues = 0
    total_lines = 0
    total_files = 0
    
    # Find all Python files
    for py_file in glob.glob(os.path.join(directory, "**/*.py"), recursive=True):
        if "__pycache__" not in py_file:
            result = analyze_python_file(py_file)
            results.append(result)
            total_files += 1
            total_lines += result.get("lines", 0)
            total_issues += len(result.get("issues", []))
    
    # Calculate overall quality score
    avg_quality = sum(r.get("quality_score", 0) for r in results) / max(len(results), 1)
    
    return {
        "directory": directory,
        "files_analyzed": total_files,
        "total_lines": total_lines,
        "total_issues": total_issues,
        "quality_score": round(avg_quality, 2),
        "file_results": results
    }


def print_summary_table(analyses: List[Dict[str, Any]]):
    """Print a summary table of analysis results"""
    print("\n" + "="*80)
    print("COMPREHENSIVE CODE ANALYSIS SUMMARY")
    print("="*80)
    print(f"{'Directory':<40} {'Files':<10} {'Issues':<10} {'Quality':<10}")
    print("-"*80)
    
    for analysis in analyses:
        dir_name = os.path.basename(analysis["directory"])
        print(f"{dir_name:<40} {analysis['files_analyzed']:<10} {analysis['total_issues']:<10} {analysis['quality_score']:<10.2f}")
    
    print("="*80)
    
    # Print detailed issues for each directory
    for analysis in analyses:
        if analysis["total_issues"] > 0:
            print(f"\nIssues in {os.path.basename(analysis['directory'])}:")
            print("-" * 40)
            issue_count = defaultdict(int)
            for file_result in analysis["file_results"]:
                for issue in file_result.get("issues", []):
                    issue_count[issue] += 1
            
            # Sort by frequency
            def get_issue_count(item):
                return item[1]
            
            for issue, count in sorted(issue_count.items(), key=get_issue_count, reverse=True)[:10]:
                print(f"  - {issue} (x{count})")


def main():
    """Main function to run analysis on specified directories"""
    directories = [
        "app/a2a/agents/calculationAgent",
        "app/a2a/agents/dataManager",
        "app/a2a/common"
    ]
    
    analyses = []
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"\nAnalyzing {directory}...")
            analysis = analyze_directory(directory)
            analyses.append(analysis)
        else:
            print(f"Directory not found: {directory}")
    
    # Print summary table
    print_summary_table(analyses)
    
    # Save detailed results to JSON
    with open("code_analysis_results.json", "w") as f:
        json.dump(analyses, f, indent=2)
    
    print(f"\nDetailed results saved to code_analysis_results.json")

if __name__ == "__main__":
    main()