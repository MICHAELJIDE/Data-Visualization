#!/usr/bin/env python
# coding: utf-8

# In[1]:


from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')

figure1 = plt.figure()
axis1 = figure1.add_subplot(projection='3d')

x =[1,7,6,3,2,4,9,8,1,9]
y =[4,6,1,8,3,7,9,1,2,4]
z =[6,4,9,2,7,8,1,3,4,9]

axis1.plot(x,y,z)

axis1.set_xlabel('X-axis')
axis1.set_ylabel('Y-axis')
axis1.set_zlabel('Z-axis')

plt.show()


# In[2]:


import seaborn as sns

plt .rcParams["figure.figsize"]=[10,8]

tips_data = sns.load_dataset('tips')

tips_data.head()


# In[3]:


bill = tips_data['total_bill'].tolist()
tip = tips_data['tip'].tolist()
size = tips_data['size'].tolist()


# In[4]:


from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')


figure2 = plt.figure()
axis2 = figure2.add_subplot(projection='3d')

axis2.plot(bill, tip, size)

axis2.set_xlabel('bill')
axis2.set_ylabel('tip')
axis2.set_zlabel('size')

plt.show()


# In[5]:


from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')


figure2 = plt.figure()
axis2 = figure2.add_subplot(projection='3d')

axis2.scatter(bill, tip, size)

axis2.set_xlabel('bill')
axis2.set_ylabel('tip')
axis2.set_zlabel('size')

plt.show()


# In[6]:


from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')

figure2 = plt.figure()
axis3 = figure2.add_subplot(projection='3d')

x3 = bill
y3 = tip
z3 = np.zeros(tips_data.shape[0])

dx = np.ones(tips_data.shape[0])
dy = np.ones(tips_data.shape[0])
dz = bill

axis3.bar3d(x3,y3,z3,dx,dy,dz)

axis3.set_xlabel('bill')
axis3.set_ylabel('tip')
axis3.set_zlabel('size')

plt.show()




# In[7]:


from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'notebook')

figure2 = plt.figure()
axis3 = figure2.add_subplot(projection='3d')

x3 = bill
y3 = tip
z3 = np.zeros(tips_data.shape[0])

dx = np.ones(tips_data.shape[0])
dy = np.ones(tips_data.shape[0])
dz = bill

axis3.bar3d(x3,y3,z3,dx,dy,dz)

axis3.set_xlabel('bill')
axis3.set_ylabel('tip')
axis3.set_zlabel('size')

plt.show()


# In[ ]:





# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = [8,6]                                                                                                             

sns.set_style("darkgrid")

titanic_data = sns.load_dataset('titanic')
titanic_data.head()


# In[9]:


from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
 

figure2 = plt.figure()
axis2 = figure2.add_subplot(projection='3d')

axis2.scatter(pclass, age, fare)

axis2.set_xlabel('pclass')
axis2.set_ylabel('age')
axis2.set_zlabel('fare')

plt.show()


# In[10]:


from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')


figure2 = plt.figure()
axis2 = figure2.add_subplot(projection='3d')

axis2.scatter (fare, age, pclass)

axis2.set_xlabel('fare')
axis2.set_ylabel('age')
axis2.set_zlabel('pclass')

plt.show()


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.figsize"] = [10,8]

titanic_data = sns.load_dataset('titanic')
titanic_data.head()


# In[12]:


from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')


figure2 = plt.figure()
axis2 = figure2.add_subplot(projection='3d')

axis2.scatter (fare, age, pclass)

axis2.set_xlabel('fare')
axis2.set_ylabel('age')
axis2.set_zlabel('pclass')

plt.show()


# In[13]:


from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')


figure2 = plt.figure()
axis3 = figure2.add_subplot(projection='3d')

axis3.scatter (fare, age, pclass)

axis3.set_xlabel('fare')
axis3.set_ylabel('age')
axis3.set_zlabel('pclass')

plt.show()


# In[14]:


from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'notebook')

figure2 = plt.figure()
axis3 = figure2.add_subplot(projection='3d')

x3 = age
y3 = fare
z3 = np.zeros(titanic_data.shape[0])

dx = np.ones(titanic_data.shape[0])
dy = np.ones(titanic_data.shape[0])
dz = pclass

axis3.bar3d(x3,y3,z3,dx,dy,dz)

axis3.set_xlabel('age')
axis3.set_ylabel('fare')
axis3.set_zlabel('pclass')

plt.show()


# In[15]:


from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'notebook')

figure3 = plt.figure()
axis3 = figure3.add_subplot(projection='3d')

x3 = age
y3 = fare
z3 = np.zeros(titanic_data.shape[0])

dx = np.ones(titanic_data.shape[0])
dy = np.ones(titanic_data.shape[0])
dz = pclass

axis3.bar3d(x3,y3,z3,dx,dy,dz)

axis3.set_xlabel('age')
axis3.set_ylabel('fare')
axis3.set_zlabel('pclass')

plt.show()


# In[17]:


from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'notebook')

figure3 = plt.figure()
axis3 = figure3.add_subplot(projection='3d')

x3 = age.reshape
y3 = fare.reshape
z3 = np.zeros(titanic_data.shape[0])

dx = np.ones(titanic_data.shape[0])
dy = np.ones(titanic_data.shape[0])
dz = pclass

axis3.bar3d(x3,y3,z3,dx,dy,dz)

axis3.set_xlabel('age')
axis3.set_ylabel('fare')
axis3.set_zlabel('pclass')


# In[18]:


from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'notebook')

figure3 = plt.figure()
axis3 = figure3.add_subplot(projection='3d')

x3 = age
y3 = fare
z3 = np.zeros(titanic_data.shape[0])

dx = np.ones(titanic_data.shape[0])
dy = np.ones(titanic_data.shape[0])
dz = pclass

axis3.bar3d(x3,y3,z3,dx,dy,dz)

axis3.set_xlabel('age')
axis3.set_ylabel('fare')
axis3.set_zlabel('pclass')

x = data[:,0].reshape(4,4)
y = data[:,1].reshape(4,4)
z = data[:,2].reshape(4,4)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X=x,Y=y,Z=z)
plt.show()


# In[19]:


from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'notebook')

figure3 = plt.figure()
axis3 = figure3.add_subplot(projection='3d')

x3 = age
y3 = fare
z3 = np.zeros(titanic_data.shape[0])

dx = np.ones(titanic_data.shape[0])
dy = np.ones(titanic_data.shape[0])
dz = pclass

axis3.bar3d(x3,y3,z3,dx,dy,dz)

axis3.set_xlabel('age')
axis3.set_ylabel('fare')
axis3.set_zlabel('pclass')

plt.show


# In[20]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = [8,6]                                                                                                             

sns.set_style("darkgrid")

titanic_data = sns.load_dataset('titanic')
pclass = titanic_data['pclass'].tolist()
age = titanic_data['age'].tolist()
fare = titanic_data['fare'].tolist()

figure4 =plt.figure()
axis4 = figure4.add_subplot(projection='3d')

axis4.scatter(pclass,age,fare)

axis4.set_xlabel('pclass')
axis3.set_ylabel('age')
axis3.set_zlabel('fare')

plt.show


# In[ ]:




