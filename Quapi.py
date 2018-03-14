import numpy as np
import cmath
import math
import time
import matplotlib.pyplot as plt
o = 1.0
e = 0.0
aux = np.array([],float)
frec = math.sqrt((o**2)+(e**2))
t_= 0.35
"""
ethak = (0.-0.j)
ethak = (0.499992-0.0000166657j)
ethak1 = (0.0100027-0.0000666627j)
ethak = (0.597816  -0.212727j)
ethak1 = (0.597816  -0.212727j)
0.333563 - 0.0353554 I
(6.10695   -2.87143 j)

Para un acoplo de 0.5 = cte
Con frec = 2.25*om
T = 0.5
    (0.063124-0.0602138j)
T = 0.6
    (0.0757489 -0.0602138j)
T = 0.75
    (0.0946861-0.0602138j)

T = 1
    (0.126248-0.0602138j)

T = 1.25
    (0.15781      -0.0602138j)
    
T = 1.5
    (0.189372      -0.0602138j)
T = 2
    0.252496 - 0.0602138 I
T = 3
    (0.378744    -0.0602138j)
T = 4
    (0.504992    -0.0602138j)
T = 5
    (0.63124    -0.0602138j)
T = 6
    (0.757489     -0.0602138j)
Los resultados reales ---> frec = 4*om
T = 0.5
    0.0985214 - 0.224727j
T = 0.6
    0.118226 - 0.224727j
T = 0.75
    (0.147782 - 0.224727j)

T = 1
    0.197043 - 0.224727j

T = 1.25
    0.246303 - 0.224727j
    
T = 1.5
    0.295564 - 0.224727j
T = 2
    0.394085 - 0.224727j
T = 3
    0.591128 - 0.224727j
T = 4
    0.788171 - 0.224727j
T = 5
    0.911591 - 0.224727j
T = 6
    (1.0141  - 0.224727j)

Para un acoplo de 0.78 = cte
T = 0.5
    0.153693 - 0.350573j
T = 0.6
    0.184432 - 0.350573j
T = 0.75
    0.23054 - 0.350573j
T = 1
    0.307387 - 0.350573j
T = 1.25
    0.3842333346973488 -0.35057346416658136j
T = 1.5
    0.46108 - 0.350573j
T = 2
    0.614773 - 0.350573j
T = 3
    0.92216 - 0.350573j)
T = 4
    1.22955 - 0.350573j
T = 5
    1.53693 - 0.350573j
T = 6
    (1.84432  - 0.350573j)

Para un acoplo de 1.5 = cte
T = 0.5
    0.295564 - 0.67418j
T = 0.6
   0.354677 - 0.67418j
T = 0.75
    0.443346 - 0.67418j
T = 1
    0.591128 - 0.67418j
T = 1.25
    0.73891 - 0.67418j
T = 1.5
    0.886692 - 0.67418j
T = 2
    1.18226 - 0.67418j
T = 3
   1.77338 - 0.67418j
T = 4
    2.36451 - 0.67418j
T = 5
    2.95564 - 0.67418j
T = 6
    3.54677 - 0.67418j

ethak =0.000588498 - 0.0000220151j
ethak1 =0.000588498 - 0.0000220151j

"""
ethak =0.000839308 - 0.00000480224j
ethak1 =0.000209827 - 0.00000480224j


class Propagador:
    """
    Construye un propagador por un AK max dado
    """
    def __init__(self,AK):
        l=aux
        self.et=np.array([])
        for ind in range(4*AK):
            l=np.append(l, [0.])
        for ind in range(2**(4*AK)):
            self.et=np.append(self.et, l)
        self.et=self.et.reshape((2**(4*AK)),4*AK)
        self.k=AK
        self.kernel = np.array([])
        self.I0 = np.array([])
        self.IK = np.array([])
        self.tensor = np.array([])
        for i in range(16):
            self.kernel=np.append(self.kernel,[0.+0.j])

        
#	ethab es la etha (k,k) y etha d es la etha (K+AK,K), en una primera fase del programa consideramos que (K+AK,K) es idéntico para todos AK         
# 	no confundir AK con AKmax anterior
    def get_et(self):
        self.cal_et()
        return self.et
    def get_tensor(self):
        self.cal_prop()
        return self.tensor
    def get_tensor_aux(self, ind):

        self.cal_prop()
        return self.tensor[ind]
    
    def get_I0(self, a, b):
        return cmath.exp(-(a-b)*(ethak*a-b*ethak.conjugate()))
                       
    def get_IK(self, a, b, c, d):
        return cmath.exp(-(c-d)*(ethak1*a-ethak1.conjugate()*b))  
    

    
                        
#	Para los kernel damos los resultados directamente, omega=o, epsilon=e.        
    def get_kernel(self, c, d, a, b):
        """
        LOS CALCULOS DE COSENOS SE HACEN RADIANES
        """
        global t
        
        t=math.radians(t_)        
        if a == 0 and b == 0:
            if c == 0 and d == 0:
                return complex((math.cos(t*frec)**2)+((e**2)*(math.sin(t*frec)**2)/(frec**2)),0)
            elif c == 0 and d == 1:
                return complex((e*o*(math.sin(t*frec))**2)/(frec**2),(o*(math.sin(t*frec))*(math.cos(t*frec))/(frec)))   
            elif c == 1 and d == 0:
                return complex((e*o*(math.sin(t*frec))**2)/(frec**2),-(o*(math.sin(t*frec))*(math.cos(t*frec))/(frec)))           
            elif c == 1 and d == 1:
                return complex((o**2)*((math.sin(t*frec))**2)/(frec**2),0)
            
                        
        elif a == 0 and b == 1:
            if c == 0 and d == 0:
                return complex((e*o*(math.sin(t*frec))**2/(frec**2)),((o*(math.sin(t*frec))*(math.cos(t*frec))/(frec))))           
            elif c == 0 and d == 1:
                return complex(((math.cos(t*frec))**2-((e**2)*(math.sin(t*frec))**2)/(frec**2)),(-(2*e)*(math.sin(t*frec)*math.cos(t*frec))/frec))            
            elif c == 1 and d == 0:
                return complex(((o**2)*(math.sin(t*frec)**2)/(frec**2)),0)                        
            elif c == 1 and d == 1:
                return complex((-(o*e)*(math.sin(t*frec)**2)/(frec**2)),(-(o*(math.sin(t*frec))*(math.cos(t*frec))/(frec))))
                      
 
        elif a == 1 and b == 0:                      
                                
            if c == 0 and d == 0:
                return complex((((o*e)*(math.sin(t*frec))**2)/(frec**2)),(-(o*(math.sin(t*frec))*(math.cos(t*frec))/(frec))))
            elif c == 0 and d == 1:
                return complex((((o**2)*(math.sin(t*frec))**2)/(frec**2)),0)
            elif c == 1 and d == 0:
                return complex(((math.cos(t*frec))**2-((e**2)*(math.sin(t*frec))**2)/(frec**2)),(2*e*(math.sin(t*frec)*math.cos(t*frec))/frec))
            elif c == 1 and d == 1:
                return complex((-((o*e)*(math.sin(t*frec))**2)/(frec**2)),((o*(math.sin(t*frec))*(math.cos(t*frec))/frec)))
                        
        elif a == 1 and b == 1:
            if c == 0 and d == 0:
                return complex(((o**2)*(math.sin(t*frec))**2)/(frec**2),0)
            elif c == 0 and d == 1:
                return complex(-(((o*e)*(math.sin(t*frec))**2)/(frec**2)),-(o*(math.sin(t*frec))*(math.cos(t*frec))/(frec)))
            elif c == 1 and d == 0:
                return complex(-((o*e)*(math.sin(t*frec))**2)/(frec**2),o*(math.sin(t*frec))*(math.cos(t*frec))/(frec))
            elif c == 1 and d == 1:
                return complex((math.cos(t*frec))**2+((e**2)*(math.sin(t*frec))**2)/(frec**2),0)                    

                       
    def cal_et(self):
        """
        Este metodo nos devuelve un array de etiquetas con valores de 0 o 1 para los 2^4*AKmax conjuntos (sk+-,sk+1+-,...,sk+2Ak-1+-)
        """

        for ind in range(2**(4*self.k)):
            i=0
            num = int(bin(ind)[2:])
            aux = listarNum(num)
            list_num=np.array([])
            while i < 4*self.k:
                if len(aux) < 4*self.k-i:
                    list_num=np.append(list_num, [0.])
                elif len(aux)==4*self.k-i:
                    list_num=np.append(list_num, aux)
                i=i+1
            """
            reversed_list_num = list_num[::-1]
            self.et[ind]=reversed_list_num
            """
            self.et[ind]=list_num
            



    def cal_prop(self):
        
        self.cal_et()
        #Esto nos actualiza los valores de self.et, realmente es una matriz donde cada fila es una combinacion de etiquetas
        
        
        for i in range(2**(4*self.k)):
            prop = 1.+0.j
            for k in range(0, 2*self.k, 2):
                l = 0
                product = 1.+0.j
                
                for j in range(self.k):
                    product = product * self.get_IK(self.et[i,k], self.et[i,k+1], self.et[i,k+2+l], self.et[i,k+3+l])
                    l = l + 2
            
                prop = prop * self.get_kernel(self.et[i,k], self.et[i,k+1], self.et[i,k+2], self.et[i,k+3])*self.get_I0(self.et[i,k], self.et[i,k+1])*product
            
            self.tensor=np.append(self.tensor, [prop])
            

            
            
        """
        i=0
        prop = 1.
        
        product = 1.
        k = 0
        for j in range(self.k):
            product = product * self.get_IK(self.et[i,k], self.et[i,k+1], self.et[i,k+2+j], self.et[i,k+3+j])
        prop = prop * self.get_kernel(self.et[i,k], self.et[i,k+1], self.et[i,k+2], self.et[i,k+3])*self.get_I0(self.et[i,k], self.et[i,k+1])*product
        k = 1
        product = 1.
        for j in range(self.k):
            product = product * self.get_IK(self.et[i,k+1], self.et[i,k+2], self.et[i,k+3+2*j], self.et[i,k+4+2*j])
        prop = prop * self.get_kernel(self.et[i,k+1], self.et[i,k+2], self.et[i,k+3], self.et[i,k+5])*self.get_I0(self.et[i,k+1], self.et[i,k+2])*product
        self.tensor=np.append(self.tensor, [prop])
        """
        

def listarNum(num):
    """
    Entra un numero y devuelve sus digitos en un array
    """
    num=str(num)
    list_num=np.array([])
    for n in num:
        n=float(n)
        list_num=np.append(list_num, n)
    return list_num


def numListar(lista):
    """
    Entra un array y devuelve un numero
    """
    num_list=str()
    for num in lista:
        num = int(num)
        num_list=num_list+str(num)
    return num_list






            
class Vector:
    #   C1 y C2 probabilidades de estar en estado fundamental y excitado

    def __init__(self,AK,ini):
        
        global my_prop
        my_prop = Propagador(AK)
        self.prop = my_prop.get_tensor()

        l=aux
        self.et=np.array([])
        for ind in range(2*AK):
            l=np.append(l, [0.])
        for ind in range(4**AK):
            self.et=np.append(self.et, l)
        self.et=self.et.reshape(4**AK,2*AK)
            
        self.k = AK
        self.vector = np.array(ini)
        self.prop_et = np.array([[]])
        self.density = np.array([0.,0.,0.,0.],complex)
    
    def cal_et(self):
        for ind in range(4**self.k):
            i=0
            num = int(bin(ind)[2:])
            aux = listarNum(num)
            list_num=np.array([])
            
            
            while i < 2*self.k:
                if len(aux) < 2*self.k-i:
                    list_num=np.append(list_num, [0.])
                elif len(aux)==2*self.k-i:
                    list_num=np.append(list_num, aux)
                i=i+1
            self.et[ind]=list_num
    

       
    def cal_A(self):
        
        global dataset1, dataset2, time, dataset3, sig_3
        time = np.array([])
        dataset1=np.array([])
        dataset2=np.array([])
        dataset3=np.array([])
        te = 0.
        

        #Inicializamos la lista del vector A inicial
        list_a = np.array(self.cal_list_a(), int)
        #Calculamos la lista de cada configuración del tensor de propagación
        for aux in self.et:        
            self.prop_et = np.append(self.prop_et, [self.cal_list_p(aux)])

        self.prop_et = np.array(self.prop_et, int)
        vector_aux = np.array(self.vector)
        suma = complex(0,0)
        
        for n in range(100000):
            
            for i in list_a:
            
            
                for j in list_a:
                    suma = suma + vector_aux[j]*self.prop[self.prop_et[j+(4**self.k)*i]]
                    """
                    print(self.prop[self.prop_et[j+(4**self.k)*i]])
                    """
                
                self.vector[i] = suma
                suma = complex(0,0)
                
            vector_aux = np.array(self.vector)  
            
            te = te+t*self.k
            dataset1 = np.append(dataset1,self.vector[0].real)
            dataset2 = np.append(dataset2,self.vector[3*(4**(self.k-1))].real)
            
            time = np.append(time, [te])
            
        #   Esencialmente obtenemos el ultimo valor actualizado de sig_3, el sig_3 estacionario    
        

    def cal_list_a(self):
        """
        Metodo accesorio de cal_A
        """
        
        list_a = np.array([])
        
        self.cal_et()
        #Para cada vector configuración binaria que etiqueta al elemento del vector A obtenemos un numero decimal sobre el número de elemento, a partir de aqui podemos hacer el calculo de la propagacion
        list_a = np.array([])
        for binn in self.et:
            dec_a = numListar(binn)
            dec_a = int(str(dec_a), 2)
            list_a = np.append(list_a, [dec_a])   
        return list_a
        
    def cal_list_p(self, aux):
        """
        Metodo accesorio de cal_A
        """
        list_p = np.array([])
        
        self.cal_et()
        #Para cada vector configuración binaria que etiqueta al elemento del vector A obtenemos un numero decimal sobre el número de elemento, a partir de aqui podemos hacer el calculo de la propagacion
        for binn in self.et:
            binn_2 = np.append(binn,aux)
            dec_p = numListar(binn_2)
            dec_p = int(str(dec_p), 2)
            list_p = np.append(list_p, [dec_p])
        return list_p

    def cal_density(self):
 
        k = 0
        for ind in range(4):
            self.density[ind] = self.vector[k]* my_prop.get_I0(self.et[k,0],self.et[k,1])
            k = k+4**(dim-1)
        
  
            
    def get_density(self):
        self.cal_density()
        return self.density
   
    def get_vector(self):
        return self.vector

    def get_prop(self):
        return self.prop

    def get_et(self):
        self.cal_et()
        return self.et
    
    def get_prop_et(self):
        self.cal_A()
        return self.prop_et
    
    def pruebas(self):
        return self.prop_et[4032]


def eigen(array):
    global aut, vals, vecs
    vals, vecs = np.linalg.eig(array)
    """
    ind=3
    aut = np.array([vecs[0,ind],vecs[1,ind],vecs[2,ind],vecs[3,ind]])
    """

def unitaria(matrix, dim):
    
    matrixcon=np.conjugate(matrix)
    matrixcon=matrixcon.transpose()
    traza=np.trace(np.dot(matrix, matrixcon)-np.identity(dim,complex))
    print(traza)
    
"""   
def timer():
    my_vector.cal_A()

my_prop=Propagador(2)
print(my_prop.get_et())
print(my_prop.get_kernel(0,0 ,0 ,0 )*my_prop.get_kernel(0,0 ,0 ,1)*my_prop.get_kernel(0,1 ,0 ,0))
print(my_prop.get_IK(1, 0, 0, 1))
print(my_prop.get_I0(0, 1))
print(my_prop.get_tensor_aux(64))
#Parece que ahora el metodo de tensores es correcto
print(np.trace(vector))
print(np.conjugate(vector).transpose())


"""
dime=2
my_prop=Propagador(dime)
lista=my_prop.get_et()
print(len(my_prop.get_et()))
print(lista)

matriz=my_prop.get_tensor()
matriz=matriz.reshape(4**dime,4**dime)

mac3=np.array([[0.,0.],[0.,0.]],complex)
mac2 = np.array([],complex)
for ind in range(4**(dime)):
    mac2 = np.append(mac2,[0.+0.j])


eigen(matriz)
print(vals)
vecs=vecs.transpose()

mac=vecs[2]

k = 0
for ind in range(4):
    mac2[ind] = mac[k]* my_prop.get_I0(lista[k,0],lista[k,1])
    k = k+4**(dime-1)

mac = mac.reshape(2**dime,2**dime)
mac2 = mac2.reshape(2**dime,2**dime)

mac3[0,0]=mac2[0,0]
mac3[0,1]=mac2[0,1]
mac3[1,0]=mac2[0,2]
mac3[1,1]=mac2[0,3]

d = 1/np.trace(mac3)
mac3=mac3*d
print(mac3)

#El método de autovalores si funciona para Akmax > 2

x=np.array([[0., 1.], [1., 0.]], complex)
y=np.array([[1., 0.], [0., -1.]], complex)
deltay=np.dot(mac3,y)
deltayp=np.trace(deltay)
deltax=np.dot(mac3,x)
deltaxp=np.trace(deltax)

print(x)
print(deltax)
print(deltaxp)
print(y)
print(deltayp)



"""


u=np.array([[0.5, 0.5, -0.5, -0.5],[0.5, 0.5,0.5,0.5],[0.5,-0.5,0.5,-0.5],[0.5,-0.5,-0.5,0.5]],float)

aut=np.dot(u, aut)


my_prop=Propagador(1)

a = my_prop.get_tensor()

a = a.reshape(4,4)


eigen(a)
aut = aut.reshape(2,2)

d = 1/np.trace(aut)

aut = aut*d

x=np.array([[0., 1.], [1., 0.]], complex)
deltax=np.dot(aut,x)
deltaxp=np.trace(deltax)
y=np.array([[1., 0.], [0., -1.]], complex)
deltay=np.dot(aut,y)
deltayp=np.trace(deltay)
print(vals)
print(aut)
print(x)
print(deltax)
print(deltaxp)
print(y)
print(deltayp)






dim = 1

num = 1**(dim-1)
inicial = np.array([])

for ind in range(num):    
    inicial = np.append(inicial, [1+0j])
    
for ind in range(4**dim-num):
    inicial = np.append(inicial, [0+0j])


my_prop=Propagador(2)

print(my_prop.get_tensor())

print(my_prop.get_I0(0,1))


my_vector = Vector(dim,inicial)
my_vector.cal_A()



dens = my_vector.get_density()
dens = dens.reshape(2,2)


print(dens)
traza=np.trace(dens)
print(traza)

dens=dens/traza
print(dens)

x=np.array([[0., 1.], [1., 0.]], complex)
y=np.array([[1., 0.], [0., -1.]], complex)
deltay=np.dot(dens,y)
deltayp=np.trace(deltay)
deltax=np.dot(dens,x)
deltaxp=np.trace(deltax)

print(x)
print(deltax)
print(deltaxp/2)
print(y)
print(deltayp)


plt.plot(time,dataset1)
plt.xlabel("tiempo")
plt.ylabel("C1")
plt.show()    

plt.plot(time,dataset2)
plt.xlabel("tiempo")
plt.ylabel("C2")
plt.show()





d = 1/np.trace(aut)
aut = aut*d
print(aut)

my_prop=Propagador(1)
a = my_prop.get_tensor()
a = a.reshape(4,4)

print(vals)
print(vecs)
eigen(a)

d = 1/np.trace(aut)
aut = aut*d
aut = aut.reshape(2,2)




aut = aut.reshape(2,2)
d = 1/np.trace(aut)
aut = aut*d
print(vals)
print(aut)
print(aut[1,1]-0.5)

print(sig_3)
plt.plot(time,dataset3)
plt.xlabel("tiempo")
plt.ylabel("<o3>/2")
plt.show()



plt.plot(time,dataset1)
plt.xlabel("tiempo")
plt.ylabel("C1")
plt.show()


"""
















