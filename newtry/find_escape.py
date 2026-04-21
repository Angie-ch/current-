import sys

with open(r"J:/用所选项目新建的文件夹/newtry/evaluate_multi.py", "rb") as f:
    raw = f.read()

print(f"File size: {len(raw)} bytes")
print(f"Bytes 430-460: {raw[430:460]}")
print()

for i in range(len(raw)):
    b = raw[i]
    if b == 92:  # backslash
        if i + 1 < len(raw):
            nb = raw[i + 1]
            if nb in (85, 117):  # U or u
                start = max(0, i - 30)
                end = min(len(raw), i + 30)
                print(f"Position {i}: backslash+{chr(nb)}")
                print(f"  Context: {raw[start:end]}")
                print()
