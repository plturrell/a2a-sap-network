#!/usr/bin/env python3
"""
Python Documentation Generator for A2A Platform
Generates comprehensive documentation from Python code, docstrings, and type hints
"""

import ast
import inspect
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import argparse
import logging
from datetime import datetime


@dataclass
class DocItem:
    name: str
    type: str
    description: str
    file_path: str
    line_number: int
    signature: Optional[str] = None
    parameters: Optional[List[Dict[str, Any]]] = None
    return_type: Optional[str] = None
    decorators: Optional[List[str]] = None
    examples: Optional[List[str]] = None


@dataclass
class ClassDoc:
    name: str
    description: str
    file_path: str
    line_number: int
    methods: List[DocItem]
    properties: List[DocItem]
    inheritance: List[str]
    decorators: List[str]


@dataclass
class ModuleDoc:
    name: str
    description: str
    file_path: str
    classes: List[ClassDoc]
    functions: List[DocItem]
    constants: List[DocItem]
    imports: List[str]


class PythonDocGenerator:
    """Advanced Python documentation generator with AST parsing"""
    
    def __init__(self, root_dir: str, output_dir: str = "./docs/python"):
        self.root_dir = Path(root_dir)
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)
        self.modules: List[ModuleDoc] = []
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_documentation(self) -> Dict[str, Any]:
        """Generate comprehensive Python documentation"""
        self.logger.info(f"Generating Python documentation for: {self.root_dir}")
        
        # Find all Python files
        python_files = list(self.root_dir.rglob("*.py"))
        self.logger.info(f"Found {len(python_files)} Python files")
        
        # Process each Python file
        for py_file in python_files:
            try:
                if self._should_skip_file(py_file):
                    continue
                    
                module_doc = self._analyze_python_file(py_file)
                if module_doc:
                    self.modules.append(module_doc)
                    
            except Exception as e:
                self.logger.warning(f"Failed to process {py_file}: {e}")
        
        # Generate documentation files
        self._generate_module_docs()
        self._generate_api_reference()
        self._generate_class_hierarchy()
        self._generate_navigation_index()
        
        stats = {
            'total_modules': len(self.modules),
            'total_classes': sum(len(m.classes) for m in self.modules),
            'total_functions': sum(len(m.functions) for m in self.modules),
            'output_directory': str(self.output_dir),
            'generated_at': datetime.now().isoformat()
        }
        
        self.logger.info(f"Documentation generated: {stats}")
        return stats
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped"""
        skip_patterns = [
            '__pycache__',
            '.git',
            'node_modules',
            'venv',
            '.env',
            'test_',
            '_test.py',
            'conftest.py'
        ]
        
        return any(pattern in str(file_path) for pattern in skip_patterns)
    
    def _analyze_python_file(self, file_path: Path) -> Optional[ModuleDoc]:
        """Analyze a Python file using AST"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content, filename=str(file_path))
            
            # Extract module information
            module_name = self._get_module_name(file_path)
            module_docstring = ast.get_docstring(tree) or ""
            
            # Analyze AST nodes
            classes = []
            functions = []
            constants = []
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_doc = self._analyze_class(node, file_path, content)
                    classes.append(class_doc)
                    
                elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                    func_doc = self._analyze_function(node, file_path, content)
                    functions.append(func_doc)
                    
                elif isinstance(node, ast.Assign) and node.col_offset == 0:
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.isupper():
                            const_doc = self._analyze_constant(node, target, file_path, content)
                            constants.append(const_doc)
                            
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_info = self._analyze_import(node)
                    imports.append(import_info)
            
            return ModuleDoc(
                name=module_name,
                description=module_docstring,
                file_path=str(file_path.relative_to(self.root_dir)),
                classes=classes,
                functions=functions,
                constants=constants,
                imports=imports
            )
            
        except Exception as e:
            self.logger.error(f"Failed to analyze {file_path}: {e}")
            return None
    
    def _get_module_name(self, file_path: Path) -> str:
        """Get module name from file path"""
        relative_path = file_path.relative_to(self.root_dir)
        return str(relative_path.with_suffix('').as_posix().replace('/', '.'))
    
    def _analyze_class(self, node: ast.ClassDef, file_path: Path, content: str) -> ClassDoc:
        """Analyze a class definition"""
        docstring = ast.get_docstring(node) or ""
        
        # Get inheritance
        inheritance = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                inheritance.append(base.id)
            elif isinstance(base, ast.Attribute):
                inheritance.append(f"{base.value.id}.{base.attr}")
        
        # Get decorators
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                decorators.append(f"{decorator.value.id}.{decorator.attr}")
        
        # Analyze methods and properties
        methods = []
        properties = []
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if any(isinstance(d, ast.Name) and d.id == 'property' for d in item.decorator_list):
                    prop_doc = self._analyze_function(item, file_path, content, is_method=True)
                    prop_doc.type = "property"
                    properties.append(prop_doc)
                else:
                    method_doc = self._analyze_function(item, file_path, content, is_method=True)
                    methods.append(method_doc)
        
        return ClassDoc(
            name=node.name,
            description=docstring,
            file_path=str(file_path.relative_to(self.root_dir)),
            line_number=node.lineno,
            methods=methods,
            properties=properties,
            inheritance=inheritance,
            decorators=decorators
        )
    
    def _analyze_function(self, node: ast.FunctionDef, file_path: Path, content: str, is_method: bool = False) -> DocItem:
        """Analyze a function or method definition"""
        docstring = ast.get_docstring(node) or ""
        
        # Parse docstring for parameters and return type
        parameters = []
        return_info = ""
        examples = []
        
        if docstring:
            parameters, return_info, examples = self._parse_docstring(docstring)
        
        # Get function signature
        signature = self._get_function_signature(node)
        
        # Get decorators
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                decorators.append(f"{decorator.value.id}.{decorator.attr}")
        
        return DocItem(
            name=node.name,
            type="method" if is_method else "function",
            description=docstring,
            file_path=str(file_path.relative_to(self.root_dir)),
            line_number=node.lineno,
            signature=signature,
            parameters=parameters,
            return_type=return_info,
            decorators=decorators,
            examples=examples
        )
    
    def _analyze_constant(self, node: ast.Assign, target: ast.Name, file_path: Path, content: str) -> DocItem:
        """Analyze a constant definition"""
        # Try to get the value
        value = ""
        try:
            if isinstance(node.value, ast.Constant):
                value = repr(node.value.value)
            elif isinstance(node.value, ast.Str):
                value = repr(node.value.s)
            elif isinstance(node.value, ast.Num):
                value = str(node.value.n)
        except:
            pass
        
        return DocItem(
            name=target.id,
            type="constant",
            description=f"Value: {value}",
            file_path=str(file_path.relative_to(self.root_dir)),
            line_number=node.lineno,
            signature=f"{target.id} = {value}"
        )
    
    def _analyze_import(self, node: Union[ast.Import, ast.ImportFrom]) -> str:
        """Analyze import statement"""
        if isinstance(node, ast.Import):
            return ", ".join(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = ", ".join(alias.name for alias in node.names)
            return f"from {module} import {names}"
        return ""
    
    def _get_function_signature(self, node: ast.FunctionDef) -> str:
        """Extract function signature"""
        args = []
        
        # Regular arguments
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {self._ast_to_string(arg.annotation)}"
            args.append(arg_str)
        
        # Default values
        defaults = node.args.defaults
        if defaults:
            for i, default in enumerate(defaults):
                arg_index = len(node.args.args) - len(defaults) + i
                if arg_index < len(args):
                    args[arg_index] += f" = {self._ast_to_string(default)}"
        
        # Varargs and kwargs
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")
        
        signature = f"{node.name}({', '.join(args)})"
        
        # Return annotation
        if node.returns:
            signature += f" -> {self._ast_to_string(node.returns)}"
        
        return signature
    
    def _ast_to_string(self, node: ast.AST) -> str:
        """Convert AST node to string representation"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Attribute):
            return f"{self._ast_to_string(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            return f"{self._ast_to_string(node.value)}[{self._ast_to_string(node.slice)}]"
        else:
            return str(node)
    
    def _parse_docstring(self, docstring: str) -> tuple:
        """Parse docstring to extract parameters, return type, and examples"""
        parameters = []
        return_info = ""
        examples = []
        
        lines = docstring.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            # Check for sections
            if line.lower().startswith('args:') or line.lower().startswith('parameters:'):
                current_section = 'params'
                continue
            elif line.lower().startswith('returns:'):
                current_section = 'returns'
                continue
            elif line.lower().startswith('example'):
                current_section = 'examples'
                continue
            elif line.lower().startswith('raises:'):
                current_section = 'raises'
                continue
            
            # Parse content based on section
            if current_section == 'params':
                param_match = re.match(r'\s*(\w+)\s*\(([^)]+)\):\s*(.*)', line)
                if param_match:
                    parameters.append({
                        'name': param_match.group(1),
                        'type': param_match.group(2),
                        'description': param_match.group(3)
                    })
                elif line.startswith('-'):
                    param_match = re.match(r'\s*-\s*(\w+):\s*(.*)', line)
                    if param_match:
                        parameters.append({
                            'name': param_match.group(1),
                            'type': 'Any',
                            'description': param_match.group(2)
                        })
            
            elif current_section == 'returns' and line:
                return_info = line
                
            elif current_section == 'examples' and line:
                examples.append(line)
        
        return parameters, return_info, examples
    
    def _generate_module_docs(self):
        """Generate individual module documentation files"""
        modules_dir = self.output_dir / "modules"
        modules_dir.mkdir(exist_ok=True)
        
        for module in self.modules:
            # Create module documentation
            content = self._format_module_doc(module)
            
            # Write to file
            module_file = modules_dir / f"{module.name.replace('.', '_')}.md"
            with open(module_file, 'w', encoding='utf-8') as f:
                f.write(content)
    
    def _format_module_doc(self, module: ModuleDoc) -> str:
        """Format module documentation as Markdown"""
        content = f"""# {module.name}

{module.description}

**File:** `{module.file_path}`

## Overview
- **Classes:** {len(module.classes)}
- **Functions:** {len(module.functions)}
- **Constants:** {len(module.constants)}

"""
        
        # Classes section
        if module.classes:
            content += "## Classes\n\n"
            for cls in module.classes:
                content += self._format_class_doc(cls)
        
        # Functions section
        if module.functions:
            content += "## Functions\n\n"
            for func in module.functions:
                content += self._format_function_doc(func)
        
        # Constants section
        if module.constants:
            content += "## Constants\n\n"
            for const in module.constants:
                content += f"### {const.name}\n"
                content += f"{const.description}\n\n"
                content += f"```python\n{const.signature}\n```\n\n"
        
        # Imports section
        if module.imports:
            content += "## Dependencies\n\n"
            content += "```python\n"
            content += "\n".join(module.imports)
            content += "\n```\n\n"
        
        return content
    
    def _format_class_doc(self, cls: ClassDoc) -> str:
        """Format class documentation"""
        content = f"### {cls.name}\n\n"
        content += f"{cls.description}\n\n"
        
        if cls.inheritance:
            content += f"**Inherits from:** {', '.join(cls.inheritance)}\n\n"
        
        if cls.decorators:
            content += f"**Decorators:** {', '.join(cls.decorators)}\n\n"
        
        # Methods
        if cls.methods:
            content += "#### Methods\n\n"
            for method in cls.methods:
                content += self._format_function_doc(method, indent="##### ")
        
        # Properties
        if cls.properties:
            content += "#### Properties\n\n"
            for prop in cls.properties:
                content += f"##### {prop.name}\n"
                content += f"{prop.description}\n\n"
        
        content += "---\n\n"
        return content
    
    def _format_function_doc(self, func: DocItem, indent: str = "### ") -> str:
        """Format function documentation"""
        content = f"{indent}{func.name}\n\n"
        content += f"{func.description}\n\n"
        
        if func.signature:
            content += f"```python\n{func.signature}\n```\n\n"
        
        if func.parameters:
            content += "**Parameters:**\n"
            for param in func.parameters:
                content += f"- `{param['name']}` ({param.get('type', 'Any')}): {param.get('description', '')}\n"
            content += "\n"
        
        if func.return_type:
            content += f"**Returns:** {func.return_type}\n\n"
        
        if func.decorators:
            content += f"**Decorators:** {', '.join(func.decorators)}\n\n"
        
        if func.examples:
            content += "**Examples:**\n"
            content += "```python\n"
            content += "\n".join(func.examples)
            content += "\n```\n\n"
        
        content += "---\n\n"
        return content
    
    def _generate_api_reference(self):
        """Generate API reference document"""
        content = f"""# API Reference

> Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Modules Overview

"""
        
        for module in self.modules:
            content += f"### [{module.name}](./modules/{module.name.replace('.', '_')}.md)\n"
            content += f"{module.description[:100]}...\n\n"
            content += f"- **Classes:** {len(module.classes)}\n"
            content += f"- **Functions:** {len(module.functions)}\n"
            content += f"- **File:** `{module.file_path}`\n\n"
        
        # Write API reference
        with open(self.output_dir / "api-reference.md", 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _generate_class_hierarchy(self):
        """Generate class hierarchy documentation"""
        content = "# Class Hierarchy\n\n"
        
        # Build inheritance tree
        all_classes = []
        for module in self.modules:
            for cls in module.classes:
                all_classes.append((module.name, cls))
        
        # Group by inheritance
        hierarchy = {}
        for module_name, cls in all_classes:
            for base in cls.inheritance:
                if base not in hierarchy:
                    hierarchy[base] = []
                hierarchy[base].append((module_name, cls.name))
        
        # Generate hierarchy documentation
        for base, derived in hierarchy.items():
            content += f"## {base}\n\n"
            for module_name, class_name in derived:
                content += f"- [{module_name}.{class_name}](./modules/{module_name.replace('.', '_')}.md#{class_name.lower()})\n"
            content += "\n"
        
        # Write class hierarchy
        with open(self.output_dir / "class-hierarchy.md", 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _generate_navigation_index(self):
        """Generate navigation index"""
        content = f"""# Python Documentation

> Auto-generated documentation for Python codebase

## 📚 Documentation Sections

- [API Reference](./api-reference.md) - Complete API documentation
- [Class Hierarchy](./class-hierarchy.md) - Class inheritance structure
- [Modules](./modules/) - Individual module documentation

## 📊 Statistics

- **Total modules:** {len(self.modules)}
- **Total classes:** {sum(len(m.classes) for m in self.modules)}
- **Total functions:** {sum(len(m.functions) for m in self.modules)}
- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🗂️ Module Index

"""
        
        for module in sorted(self.modules, key=lambda m: m.name):
            content += f"- [{module.name}](./modules/{module.name.replace('.', '_')}.md)\n"
        
        # Write navigation index
        with open(self.output_dir / "README.md", 'w', encoding='utf-8') as f:
            f.write(content)


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive Python documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('root_dir', nargs='?', default='.', 
                       help='Root directory to analyze (default: current directory)')
    parser.add_argument('--output', '-o', default='./docs/python',
                       help='Output directory for documentation (default: ./docs/python)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Set logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        generator = PythonDocGenerator(args.root_dir, args.output)
        stats = generator.generate_documentation()
        
        print(f"\n✅ Python documentation generated successfully!")
        print(f"📁 Output directory: {stats['output_directory']}")
        print(f"📊 Modules documented: {stats['total_modules']}")
        print(f"🏗️ Classes documented: {stats['total_classes']}")
        print(f"🔧 Functions documented: {stats['total_functions']}")
        
    except Exception as e:
        logging.error(f"Documentation generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()