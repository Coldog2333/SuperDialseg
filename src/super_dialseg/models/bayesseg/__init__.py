# check whether bayesseg is installed, if not, install it
import os
import subprocess

bayesseg_module_dirpath = os.path.dirname(__file__)

CLASSPATH="classes:lib/colt.jar:lib/lingpipe-3.4.0.jar:lib/MinCutSeg.jar:lib/mtj.jar:lib/options.jar:lib/log4j-1.2.14.jar"
CONFIGPATH=os.path.join(bayesseg_module_dirpath, "config/dp.config")

# if classes is empty, compile it
if not os.path.exists(os.path.join(bayesseg_module_dirpath, 'classes')):
    os.mkdir(os.path.join(bayesseg_module_dirpath, 'classes'))

if os.listdir(os.path.join(bayesseg_module_dirpath, 'classes')) == []:
    subprocess.call('ant -buildfile %s' % os.path.join(bayesseg_module_dirpath, 'build.xml'), shell=True)

subprocess.call('chmod 777 %s' % os.path.join(bayesseg_module_dirpath, 'segment'), shell=True)


def rewrite_dp_config(dp_config_path):
    with open(dp_config_path, 'r', encoding='utf-8') as f:
        content = f.read()

    content = content.replace(
        'stop-words=config/STOPWORD.list',
        'stop-words=%s' % os.path.join(bayesseg_module_dirpath, 'config/STOPWORD.list'))
    content = content.replace(
        'cuephrase-file=config/CUEPHRASES.hl',
        'cuephrase-file=%s' % os.path.join(bayesseg_module_dirpath, 'config/CUEPHRASES.hl'))

    with open(dp_config_path, 'w', encoding='utf-8') as f:
        f.write(content)

rewrite_dp_config(dp_config_path=CONFIGPATH)
