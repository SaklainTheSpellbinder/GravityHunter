$report = "GravityHunter_Diagnostics.txt"

# Start fresh
"" | Out-File $report

"============================================================" | Add-Content $report
"GRAVITYHUNTER COMPLETE DIAGNOSTIC REPORT" | Add-Content $report
"Generated: $(Get-Date)" | Add-Content $report
"============================================================" | Add-Content $report


"`n`n============================================================" | Add-Content $report
"1. PYTHON INFORMATION" | Add-Content $report
"============================================================" | Add-Content $report

python --version 2>&1 | Out-File $report -Append
python -c "import sys; print('Executable:', sys.executable); print('Version:', sys.version)" 2>&1 |
    Out-File $report -Append


"`n`n============================================================" | Add-Content $report
"2. IMPORTANT PACKAGE VERSIONS" | Add-Content $report
"============================================================" | Add-Content $report

python -c "
import importlib

packages = [
    'numpy',
    'scipy',
    'h5py',
    'plotly',
    'streamlit',
    'pytest'
]

for p in packages:
    try:
        m = importlib.import_module(p)
        print(f'{p}: {getattr(m, ""__version__"", ""unknown"")}')
    except Exception as e:
        print(f'{p}: ERROR - {e}')
" 2>&1 | Out-File $report -Append


"`n`n============================================================" | Add-Content $report
"3. PYRIGHT / TYPE CHECKING" | Add-Content $report
"============================================================" | Add-Content $report

python -m pyright . 2>&1 |
    Out-File $report -Append


"`n`n============================================================" | Add-Content $report
"4. PYTHON COMPILE CHECK" | Add-Content $report
"============================================================" | Add-Content $report

python -m compileall `
    gravityhunter `
    tests `
    scripts `
    examples `
    streamlit_app.py `
    2>&1 | Out-File $report -Append


"`n`n============================================================" | Add-Content $report
"5. PYTEST FULL TEST SUITE" | Add-Content $report
"============================================================" | Add-Content $report

python -m pytest -vv 2>&1 |
    Out-File $report -Append


"`n`n============================================================" | Add-Content $report
"6. GIT STATUS" | Add-Content $report
"============================================================" | Add-Content $report

git status --short 2>&1 |
    Out-File $report -Append


"`n`n============================================================" | Add-Content $report
"7. GIT DIFF SUMMARY" | Add-Content $report
"============================================================" | Add-Content $report

git diff --stat 2>&1 |
    Out-File $report -Append


"`n`n============================================================" | Add-Content $report
"END OF REPORT" | Add-Content $report
"============================================================" | Add-Content $report

Write-Host ""
Write-Host "Diagnostic report created:"
Write-Host $report