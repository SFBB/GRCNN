import os




os.chdir(os.path.split(os.path.realpath(__file__))[0])

# d = 'models/'
models = os.listdir("models/")
results = os.listdir("results/")


def is_int(str):
    try:
        a = int(str)
        return True
    except:
        return False

def not_in_whitelist(dir, lst):
    if dir.split("_")[2] not in lst:
        return True
    return False

to_remove = []
whitelist = []
for model in models:
    if "_models_" in model and is_int(model.split("_")[2]) and not_in_whitelist(model, whitelist):
        to_remove.append(os.path.join("models", model))
for result in results:
    if "_results_" in result and is_int(result.split("_")[2]) and not_in_whitelist(result, whitelist):
        to_remove.append(os.path.join("results", result))


print(len(to_remove))
print(to_remove)
m = input("To remove?")

if m == "y":
    for dir in to_remove:
        os.rmdir(dir)