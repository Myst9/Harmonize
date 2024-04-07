cd Client
for /f "delims=" %%i in ('cd') do set CURRENT_DIR=%%i
start chrome --load-extension="%CURRENT_DIR%" github.com
echo "Chrome Extension Loaded!"
cd ../API
echo "Going to start Server!"
python test.py