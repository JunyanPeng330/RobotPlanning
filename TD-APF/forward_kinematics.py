from typing import List
import coordinate_transform_function as ctf
from DH_parameters import robot_DH_param, Joint_point

#Get the DH parameters of ur5
a = robot_DH_param.a
alpha = robot_DH_param.alpha
d = robot_DH_param.d
theta_o = robot_DH_param.theta_o

def FK(joint_point:Joint_point) -> List:
          theta = joint_point.theta
          #print('theta:', theta)

          T12 = ctf.rotz(theta[0]+theta_o[0])*ctf.trans(0,0,d[0])*ctf.rotx(alpha[0])*ctf.trans(a[0],0,0)
          T23 = ctf.rotz(theta[1]+theta_o[1])*ctf.trans(0,0,d[1])*ctf.rotx(alpha[1])*ctf.trans(a[1],0,0)
          T34 = ctf.rotz(theta[2]+theta_o[2])*ctf.trans(0,0,d[2])*ctf.rotx(alpha[2])*ctf.trans(a[2],0,0)
          T45 = ctf.rotz(theta[3]+theta_o[3])*ctf.trans(0,0,d[3])*ctf.rotx(alpha[3])*ctf.trans(a[3],0,0)
          T56 = ctf.rotz(theta[4]+theta_o[4])*ctf.trans(0,0,d[4])*ctf.rotx(alpha[4])*ctf.trans(a[4],0,0)
          T67 = ctf.rotz(theta[5]+theta_o[5])*ctf.trans(0,0,d[5])*ctf.rotx(alpha[5])*ctf.trans(a[5],0,0)

          T2 = T12
          T3 = T12*T23
          T4 = T12*T23*T34
          T5 = T12*T23*T34*T45
          T6 = T12*T23*T34*T45*T56
          T7 = T12*T23*T34*T45*T56*T67
          '''
          print('T2:', T2)
          print('T3:', T3)
          print('T4:', T4)
          print('T5:', T5)
          print('T6:', T6)
   
          print('T7:', T7)
          '''
          return([T2, T3 ,T4 ,T5, T6, T7])