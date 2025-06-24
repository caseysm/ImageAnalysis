import cellpose
print("Cellpose version:", cellpose.version)
print("Available attributes:", dir(cellpose))

# Try to import submodules
try:
    import cellpose.models
    print("models submodule exists, contents:", dir(cellpose.models))
except ImportError:
    print("models submodule does not exist")
    
try:
    from cellpose import CPSAM
    print("CPSAM imported from cellpose, contents:", dir(CPSAM))
except ImportError:
    print("Cannot import CPSAM from cellpose")
    
# Try other imports
try:
    from cellpose import io, CPSAM, CPSAMpredict
    print("io, CPSAM, CPSAMpredict can be imported directly")
except ImportError as e:
    print(f"Import error: {e}")