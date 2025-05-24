import os, shutil, argparse
from omegaconf import OmegaConf

from utils.FIND import FINDFrame
from utils.Logger import MyLogger, reproduc
        
def select_mode():
    mode_dict = {'eval':args.eval, 'refine':args.refine, 'top':args.top, 'sparse':args.sparse}
    if os.path.exists(args.d):
        mode_dict['mode'] = 'eval' if args.refine==0 else 'refine'
        opt_dir = os.path.join(args.d, 'scripts')
        for file in os.listdir(opt_dir):
            if '.yaml' in file:
                opt = OmegaConf.load(os.path.join(opt_dir, file))
        opt.Log.outputs_dir = os.path.dirname(args.d)
        opt.Log.project_name = os.path.basename(args.d)
        opt.Log.time, opt.Log.stdlog, opt.Log.tensorboard = False, False, False
        opt.Structure.latent.sparse = args.sparse
        legal_step = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]
        if (args.eval not in legal_step) or (args.refine not in legal_step):
            raise NotImplemented
    else:
        mode_dict['mode'] = 'train'
        opt = OmegaConf.load(args.p)
    return mode_dict, opt

def main():
    reproduc()
    mode_dict, opt = select_mode()
    log = MyLogger(**opt['Log'])
    # code backup   
    shutil.copy(args.p, log.script_dir)
    shutil.copy(__file__, log.script_dir)
    shutil.copy('utils/Samplers.py', log.script_dir)
    shutil.copy('utils/FIND.py', log.script_dir)
    # FIND: latent identify->c2f polynomial regression->symbolic regression
    frame = FINDFrame(opt=opt, log=log)
    frame.start(mode_dict=mode_dict)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='batch discovery')
    parser.add_argument('-p', type=str, default='opt/rlc.yaml', help='config file path')
    parser.add_argument('-d', type=str, default='outputs/dir', help='evaluate or refine dir')
    parser.add_argument('-eval', type=float, default=0.1, help='evaluate precision')
    parser.add_argument('-refine', type=float, default=0.0, help='refine step')
    parser.add_argument('-top', type=int, default=20, help='number of displayed coefficients')
    parser.add_argument('-sparse', type=int, default=20, help='sparsity of displayed coefficients')
    args = parser.parse_args()
    main()