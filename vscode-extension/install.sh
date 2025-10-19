#!/bin/bash

# VSCode Struckdown Extension Installer
# Usage: ./install.sh

set -e

EXTENSION_NAME="struckdown-0.1.0"
EXTENSION_DIR="$HOME/.vscode/extensions"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Installing Struckdown VSCode Extension..."
echo ""

# Create extensions directory if it doesn't exist
mkdir -p "$EXTENSION_DIR"

# Remove old version if exists
if [ -d "$EXTENSION_DIR/$EXTENSION_NAME" ]; then
    echo "Removing existing installation..."
    rm -rf "$EXTENSION_DIR/$EXTENSION_NAME"
fi

# Copy extension
echo "Copying extension files..."
cp -r "$SCRIPT_DIR" "$EXTENSION_DIR/$EXTENSION_NAME"

# Remove install script from the copied version
rm -f "$EXTENSION_DIR/$EXTENSION_NAME/install.sh"
rm -f "$EXTENSION_DIR/$EXTENSION_NAME/test-syntax.sd"

echo ""
echo "✓ Installation complete!"
echo ""
echo "Next steps:"
echo "1. Restart VSCode or run 'Developer: Reload Window' (Cmd/Ctrl+Shift+P)"
echo "2. Open any .sd file to see syntax highlighting"
echo ""
echo "To test, try opening: $SCRIPT_DIR/test-syntax.sd"
echo ""
