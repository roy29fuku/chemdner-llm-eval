import os,json,subprocess,argparse,time
p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');p.add_argument('--selection',default='runs/final-selection.json');args=p.parse_args()
os.chdir('/home/agent')
if args.wait:
    while subprocess.run(['pgrep','-f','^bash code/(pipeline|large|link).sh$'],stdout=subprocess.DEVNULL).returncode==0:time.sleep(5)
while not os.path.exists(args.selection):time.sleep(5)
selection=json.load(open(args.selection));names=selection['names']
configs=[];manifest=[]
for i,name in enumerate(names):
    root='runs/'+name
    argfile=root+'/arguments.json'
    if os.path.exists(argfile):opts=json.load(open(argfile))
    else:opts=dict(model='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',seed=17,epochs=5,lr=2e-5,batch=24,size=256,checkpoint='runs/baseline/best',typed=False)
    validation_epoch=json.load(open(root+'/best.json'))['epoch'];epoch=validation_epoch
    continued=name in selection.get('adapt_names',[])
    if continued:
        opts=opts.copy();opts.update(checkpoint=root+'/best',epochs=3,lr=8e-6);epoch=3
    config=json.load(open(root+'/best/config.json'))
    program='code/case_ner.py' if config.get('case_features') else 'code/ner.py'
    out='runs/final-'+name
    command=['python3',program,'--full','--holdout','0','--out',out,'--model',opts['model'],'--epochs',str(opts['epochs']),'--save-epoch',str(epoch),'--stop-after',str(epoch),'--seed',str(opts['seed']),'--lr',str(opts['lr']),'--batch',str(opts['batch']),'--size',str(opts['size'])]
    if opts.get('checkpoint'):command+=['--checkpoint',opts['checkpoint']]
    if opts.get('typed'):command+=['--typed']
    print('FINAL TRAIN',name,'selected epoch',epoch,flush=True)
    with open(out+'.log','w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    predout=out+'-pred'
    command=['python3',program,'--mode','predict','--predict-test-only','--checkpoint',out+'/chosen','--out',predout,'--model',opts['model'],'--batch',str(opts['batch']),'--size',str(opts['size'])]
    with open(predout+'.log','w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    configs.append(dict(model=opts['model'],run=predout))
    manifest.append(dict(validation_run=name,final_run=out,selected_epoch=epoch,validation_epoch=validation_epoch,continued=continued,validation=json.load(open(root+'/best.json')),arguments=opts,program=program))
    json.dump(configs,open('code/final-ensemble.json','w'),indent=2)
    json.dump(manifest,open('runs/final-manifest.json','w'),indent=2)
print('FINAL MODELS COMPLETE',flush=True)
