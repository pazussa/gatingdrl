#!/usr/bin/env python3

import os
import sys

# Archivos y directorios a excluir
EXCLUDED_FILES = {
  
}
EXCLUDED_DIRS = {
    '__pycache__', 
    '.git', 
    '.idea', 
    'venv', 
    'env', 
    'build', 
    'dist'
}

def collect_python_files(root_path):
    """Recopila todas las rutas de archivos Python en el proyecto"""
    py_files = []
    for root, dirs, files in os.walk(root_path):
        # Evitar directorios excluidos
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        
        # Procesar archivos Python
        for file in files:
            if file.endswith('.py') and file not in EXCLUDED_FILES:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, root_path)
                py_files.append((rel_path, full_path))
    
    # Ordenar los archivos por ruta relativa para mejor lectura
    return sorted(py_files)

def save_code_to_file(py_files, output_file='project_code.txt'):
    """Guarda el contenido de todos los archivos en un único archivo de texto"""
    with open(output_file, 'w', encoding='utf-8') as f:
        
        f.write("# ===================================\n\n")
        
        for rel_path, full_path in py_files:
            f.write(f"## ARCHIVO: {rel_path}\n")
            f.write("## " + "=" * 50 + "\n\n")
            
            try:
                with open(full_path, 'r', encoding='utf-8') as src_file:
                    content = src_file.read()
                    f.write(content)
                    
                # Añadir líneas en blanco para mejor separación
                f.write("\n\n\n")
            except Exception as e:
                f.write(f"# ERROR: No se pudo leer el archivo: {e}\n\n")
    
    print(f"Código recopilado y guardado en: {output_file}")
    return output_file

def main():
    # Obtener la ruta del proyecto (directorio actual)
    project_path = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(project_path, "project_code.txt")
    
    print(f"Recopilando código fuente del proyecto en: {project_path}")
    py_files = collect_python_files(project_path)
    print(f"Se encontraron {len(py_files)} archivos Python")
    
    saved_file = save_code_to_file(py_files, output_file)
    print(f"Proceso completado. El código se ha guardado en: {saved_file}")

if __name__ == "__main__":
    main()
