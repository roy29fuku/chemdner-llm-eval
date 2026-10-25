import json,os,pathlib,subprocess,shutil
ROOT=pathlib.Path('/workspace');cfg=json.load(open(ROOT/'results/final_config.json'));runs=cfg['final_runs']
for run in runs:
 assert pathlib.Path(run,'test_logits.npz').exists(),f'Missing completed predictions: {run}'
out=ROOT/'results/final_test.tsv'
cmd=['python3',str(ROOT/'code/combine.py')]+runs+['--constraint','both','--average',cfg['average'],'--weights']+list(map(str,cfg['weights']))+['--bias='+str(cfg['bias']),'--begin-bias='+str(cfg['begin_bias']),'--output',str(out)]
subprocess.run(cmd,check=True)
for script,suffix in [('abbreviations.py','.abbr'),('filter.py','.filtered'),('balance.py','.balanced')]:
 subprocess.run(['python3',str(ROOT/'code'/script),'test',str(out)],check=True);out=pathlib.Path(str(out)+suffix)
tmp=ROOT/'output/predictions.tsv.tmp';shutil.copyfile(out,tmp)
check=subprocess.run(['python3',str(ROOT/'check_format.py'),str(tmp)],check=True,text=True,capture_output=True);print(check.stdout,flush=True);os.replace(tmp,ROOT/'output/predictions.tsv')
summary={'validation_f1':cfg['postprocessed_development_f1'],'validation_used_training_only_base_models':True,'final_base_models_trained_on_training_and_development':True,'format_check':check.stdout.strip(),'configuration':cfg}
json.dump(summary,open(ROOT/'results/final_summary.json','w'),indent=2)
print('FINAL PREDICTIONS SAVED',flush=True)
