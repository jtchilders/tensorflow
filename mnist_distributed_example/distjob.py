#!/usr/bin/env python
import os,sys,optparse,logging,subprocess,socket,time
from mpi4py import MPI
logger = logging.getLogger(__name__)

MNISTDIST_EXE='/projects/EnergyFEC_2/anl/cooley/machinelearning/tensorflow/mnist_distributed_example/mnistdist.py'

LOG_DIR='.'

def main():
   ''' Run the distributed mnist example  '''
   logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s:%(name)s:%(message)s')

   parser = optparse.OptionParser(description='')
   parser.add_option('-x','--norun',dest='norun',help='do not run the jobs',default=False,action='store_true')
   parser.add_option('-s','--train-steps',dest='train_steps',help='number of global steps to use in the training',default=1000,type='int')
   parser.add_option('-l','--log-dir',dest='log_dir',help='directory to save the log files and checkpoints',default=LOG_DIR)
   parser.add_option('-e','--exe',dest='exe',help='mnist script',default=MNISTDIST_EXE)
   options,args = parser.parse_args()

   
   manditory_args = [
                     'train_steps',
                     'log_dir',
                  ]

   for man in manditory_args:
      if options.__dict__[man] is None:
         logger.error('Must specify option: ' + man)
         parser.print_help()
         sys.exit(-1)
   
   

   comm = MPI.COMM_WORLD
   rank = comm.Get_rank()
   nranks = comm.Get_size()

   if rank == 0:
      if not os.path.exists(options.log_dir):
         logger.info('log directory does not exist: ' + options.log_dir)
         os.makedirs(options.log_dir)
         if not os.path.exists(options.log_dir):
            raise Exception('path does not exist and could not create it: ' + options.log_dir)
   comm.Barrier()

   # get the node lists
   nodefile = os.environ['COBALT_NODEFILE']
   nodes = [f.strip() for f in file(nodefile)]

   num_param_servers = 1
   num_nodes = nranks
   num_worker_servers = num_nodes - num_param_servers

   ps_hosts = [f+':2222' for f in nodes[:num_param_servers]]
   worker_hosts = [f+':2222' for f in nodes[num_param_servers:]]
   hostname =  socket.gethostname() + '.cooley:2222'
   logger.info('hostname: ' + hostname)
   host_index=-1

   jobname=""
   if hostname in ps_hosts:
       host_index = ps_hosts.index(hostname)
       jobname = 'ps'
   elif hostname in worker_hosts:
       host_index = worker_hosts.index(hostname)
       jobname = 'worker'
   else:
       logger.info('hostname not found in hosts file %s', hostname)
  
   data_dir = '/projects/EnergyFEC_2/anl/cooley/machinelearning/tensorflow/mnist_distributed_example/data'
   cmd = ('python %s --task_index %i --sync_replicas True --ps_hosts %s --worker_hosts %s --job_name %s --data_dir %s --num_gpus %i --train_steps %i --log_dir %s' % 
            (options.exe,host_index,','.join(ps_hosts),','.join(worker_hosts),jobname,
            data_dir,nranks,options.train_steps,options.log_dir)
         )
   logger.info('rank: %i cmd: %s',rank,cmd)
   if not options.norun:
      fstdout = open(options.log_dir + '/%i.stdout.txt'%rank,'w')
      fstderr = open(options.log_dir + '/%i.stderr.txt'%rank,'w')
      p = subprocess.Popen(cmd,stdout=fstdout,stderr=fstderr,shell=True)
      while(p.poll() is None):
         run_nvidia_smi(rank)
         time.sleep(5)
      (stdout,stderr) = p.communicate()
      print 'after popen', stdout
      #for line in p.stdout:
      #   print line

      #if p.returncode != 0:
      #   print 'tensorflow exited with non-zero value:' ,p.returncode

def run_nvidia_smi(rank):
   logger.info('rank ' + str(rank) + ' running nvidia-smi')
   cmd = 'nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv'
   p = subprocess.Popen(cmd.split(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT)

   stdout,stderr = p.communicate()
   logger.info('rank ' + str(rank) + ' nvidia-smi output: ' + stdout)



if __name__ == "__main__":
   main()


