cd Client
for /f "delims=" %%i in ('cd') do set CURRENT_DIR=%%i
start chrome --load-extension="%CURRENT_DIR%"