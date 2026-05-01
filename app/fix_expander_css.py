#!/usr/bin/env python3
# Fix expander CSS to prevent clicks from reaching it

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the expander CSS section
old_pattern = '.stSidebar .stExpander { display: none !important; height: 0 !important; width: 0 !important; }'
new_pattern = '.stSidebar .stExpander { display: none !important; height: 0 !important; width: 0 !important; pointer-events: none !important; visibility: hidden !important; }'

if old_pattern in content:
    content = content.replace(old_pattern, new_pattern)
    print("Updated expander CSS with pointer-events")
else:
    print("Pattern not found, file may have been modified")

# Also add pointer-events none to expander children
if '.stSidebar .stExpander button' in content and 'pointer-events' not in content.split('.stSidebar .stExpander button')[1].split('}')[0]:
    old_btn = '.stSidebar .stExpander button { position: absolute; left: -10000px; width: 1px; height: 1px; overflow: hidden; }'
    new_btn = '.stSidebar .stExpander button { position: absolute; left: -10000px; width: 1px; height: 1px; overflow: hidden; pointer-events: none !important; }'
    content = content.replace(old_btn, new_btn)
    print("Updated expander button CSS")

# Write back
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("CSS fix applied successfully")
