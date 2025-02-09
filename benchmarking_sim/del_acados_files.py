
import os
import shutil

current_dir = os.getcwd()
print(current_dir)

# delete the folder has *_del_*
for root, dirs, files in os.walk(current_dir):
    for dir in dirs:
        if 'c_generated_code' in dir:
            # shutil.rmtree(os.path.join(root, dir))
            os.system(f'rm -rf {os.path.join(root, dir)}')
            print('delete:', os.path.join(root, dir))
    for files in files:
        if 'acados_ocp_solver' in files:
            # shutil.rmtree(os.path.join(root, dir))
            os.system(f'rm -rf {os.path.join(root, files)}')
            print('delete:', os.path.join(root, files))

# delete the folder has *_del_* using wildcard
# os.system(f'rm -rf {current_dir}/*_del_*')
