import shutil

print("***")
print("***Removing current python version***")
shutil.rmtree("buildroot-at91/package/python3", ignore_errors=True, onerror=None)
print("***Successfully removed current python version***")

print("***Installing new version***")
shutil.copytree("content_of_python_package/", "buildroot-at91/package/", dirs_exist_ok=True)
print("***Successfully changed python version***")

