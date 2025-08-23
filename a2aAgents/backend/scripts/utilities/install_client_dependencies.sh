#!/bin/bash

echo "🔧 Installing Production Client Dependencies..."
echo "==============================================="

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  Warning: No virtual environment detected. Consider using 'python -m venv venv' and 'source venv/bin/activate'"
    echo "Continuing with system Python..."
fi

echo ""
echo "📦 Installing core dependencies..."

# Install core dependencies
pip install python-dotenv>=1.0.0
pip install httpx>=0.25.0
pip install openai>=1.3.0

echo ""
echo "🌐 Installing Perplexity dependencies..."
pip install "httpx[http2]>=0.25.0"

echo ""
echo "🗄️  Installing SAP HANA dependencies..."
pip install hdbcli>=2.16.0

echo ""
echo "💾 Installing SQLite dependencies..."
pip install aiosqlite>=0.19.0

echo ""
echo "🧪 Installing testing dependencies..."
pip install pytest>=7.4.0
pip install pytest-asyncio>=0.21.0

echo ""
echo "✅ All dependencies installed successfully!"
echo ""
echo "🚀 You can now run the production client tests:"
echo "   python3 test_production_clients.py"
echo ""
echo "💡 If you encounter any import errors, ensure all dependencies are properly installed."
