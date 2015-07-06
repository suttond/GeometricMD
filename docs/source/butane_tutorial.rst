Tutorial
========

Butane Simulation
-----------------
In this tutorial we will use the MODOI package to compute the transition path for a butane molecule between it's trans
and gauche states.

Configuring MODOI for Butane
----------------------------
First copy the files x0.xyz and xN.xyz from *<modoi-install-path>/Examples/Butane/* into *<modoi-install-path>/Experiment/*.
This will describe the start and end configurations of the Butane molecule to the MODOI software.

Next we need to tell MODOI under what conditions to perform the experiment. This is achieved using a .bkhf file.
Create a blank text file called Example.bkhf in <modoi-install-path>/Experiment/. In this file write the following

.. include:: ../../Examples/ConfigurationScript/Example.bkhf
   :literal:

The parameters sn and en tell MODOI where to find the XYZ files for the start and end configurations. The parameter ln
tells MODOI how many discretisation points to use when computing the local geodesics and the parameter gn tells MODOI
how many discretisation points to use for the global geodesic. In principle, increasing ln improves the accuracy of the
final solution but is computationally expensive when large. The parameter pa corresponds to the total energy of the
mechanical system. The parameter tol determines how close to the true solution the simulation should be before stopping.

MODOI is now configured to perform a Butane simulation. Any other information the program requires is inferred from the
above information.

Running the Simulation
----------------------
To run the simulation simply start the script 'Local_Simulation.py' using the command

>>> python Local_Simulation.py

in a shell/terminal window navigated to *<modoi-install-path>/*.