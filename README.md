Hey thanks for trying out my software. Truth be said it is long from being end product but for now I hope this will do.

In Theta Beta Gamma, you can draw your own geometry and visualize how shocks would form and behave around it. You can design engine inlets, aircraft silhouettes, high speed projectiles et cetera. The workflow is pretty straight forward. 

head to TBG/dist/TBG.exe, run it.
To draw your own geometry,  to files>draw. For ease of drawing, you can overlay any image over the canvas and trace it or free hand it using splines and lines. If you are trying to design moving parts, you can use actuators. Once done hit finish. 
For quick trialing, I've made some save files. You can check them out at file>open>saves.

Once you import your geometry, head to edit>setup to manage sim conditions. And thats pretty much it. Remember the airflow is from right to left. You can turn it off to simulate your own airflow from user define source lines. Play around with mach numbers and actuators to see how shock behaves. Now this software cannot compute subsonic or transonic conditions and bow shock calculations are not that good, for instance, I'm using cylindrical assumptions even if your geometry features a wedge. But oblique shockwave calculations are pretty solid, validated it with questions done in class. 

A little WIP easter egg (don't do this unless you have a good laptop), you can visualize pressure, density, mach distributions, schlieren map if you head to view>map. I'm still working on this so take it with a grain of salt.

Thank you! You can give me suggestions which I'll try my best to implement ;)
