import gmsh
import sys
# Before using any functions in the Python API, Gmsh must be initialized:
gmsh.initialize()

# Next we add a new model named "t1" (if gmsh.model.add() is not called a new
# unnamed model will be created on the fly, if necessary):
gmsh.model.add("circ")

vp0 = -12
vp1 = 0
gaugeHeight = 15
file_number = 1
gaugeWidth = 7.5
gaugeThickness = 1.5
lc=1


p1 = gmsh.model.geo.addPoint(-gaugeWidth,gaugeHeight,0,lc) 
p2 = gmsh.model.geo.addPoint(gaugeWidth,gaugeHeight,0,lc)
p5 = gmsh.model.geo.addPoint(vp0,0,0,lc)
p7 = gmsh.model.geo.addPoint(vp1,0,0,lc)

if abs(vp0) != gaugeWidth:
  s0 = -gaugeWidth-vp0
  l0 = gaugeHeight*2
  r0 = (s0/2) + ((l0*l0)/(8*s0))
  
  p6 = gmsh.model.geo.addPoint(vp0+r0,0,0,lc)

  c1 = gmsh.model.geo.addCircleArc(p1,p6,p5)

else:
  
  c1 = gmsh.model.geo.addLine(p1,p5)
  
if abs(vp1) != gaugeWidth:
  s1 = gaugeWidth-vp1
  l1 = gaugeHeight*2
  r1 = (s1/2) + ((l1*l1)/(8*s1))
  
  
  p8 = gmsh.model.geo.addPoint(vp1+r1,0,0,lc)

  c2 = gmsh.model.geo.addCircleArc(p2,p8,p7)
else:
  c2 = gmsh.model.geo.addLine(p2,p7)

l1 = gmsh.model.geo.addLine(p1,p2)
l3 = gmsh.model.geo.addLine(p7,p5)

p9 = gmsh.model.geo.addPoint(-gaugeWidth,gaugeHeight+2,0,1)
p10 = gmsh.model.geo.addPoint(gaugeWidth,gaugeHeight+2,0,1)

l8 = gmsh.model.geo.addLine(p2,p10)
l9 = gmsh.model.geo.addLine(p10,p9)
l10 = gmsh.model.geo.addLine(p9,p1)

#p11 = gmsh.model.geo.addPoint(gaugeWidth+2.5,gaugeHeight+2+6.614378277661476,0,1)
p12 = gmsh.model.geo.addPoint(gaugeWidth+10,gaugeHeight+2,0,1)

#p13 = gmsh.model.geo.addPoint(-gaugeWidth-2.5,gaugeHeight+2+6.614378277661476,0,1)
p14 = gmsh.model.geo.addPoint(-gaugeWidth-10,gaugeHeight+2,0,1)

#c3 = gmsh.model.geo.addCircleArc(p10,p12,p11)
#c4 = gmsh.model.geo.addCircleArc(p13,p14,p9)

#l15 = gmsh.model.geo.addLine(p13,p11)

cl1 = gmsh.model.geo.addCurveLoop([l1,c2,l3,-c1])
s1 = gmsh.model.geo.addPlaneSurface([cl1])
cl2 = gmsh.model.geo.addCurveLoop([l1,l8,l9,l10])
s2 = gmsh.model.geo.addPlaneSurface([cl2])
#cl3 = gmsh.model.geo.addCurveLoop([c3,-l15,c4,-l9])
#s3 = gmsh.model.geo.addPlaneSurface([cl3])

#gmsh.model.geo.synchronize()

gmsh.model.geo.mesh.set_transfinite_curve(l1,15)
gmsh.model.geo.mesh.set_transfinite_curve(l3,15)
gmsh.model.geo.mesh.set_transfinite_curve(l9,15)

gmsh.model.geo.mesh.set_transfinite_curve(c1,15)
gmsh.model.geo.mesh.set_transfinite_curve(c2,15)
gmsh.model.geo.mesh.set_transfinite_curve(l8,6)
gmsh.model.geo.mesh.set_transfinite_curve(l10,6)

#gmsh.model.geo.mesh.set_transfinite_curve(c3,5)
#gmsh.model.geo.mesh.set_transfinite_curve(c4,5)
#gmsh.model.geo.mesh.set_transfinite_curve(l15,15)


gmsh.model.geo.mesh.set_transfinite_surface(s1)
gmsh.model.geo.mesh.set_transfinite_surface(s2)
#gmsh.model.geo.mesh.set_transfinite_surface(s3)

gmsh.model.geo.mesh.setRecombine(2,s1)
gmsh.model.geo.mesh.setRecombine(2,s2)
#gmsh.model.geo.mesh.setRecombine(2,s3)

gmsh.model.geo.synchronize()

#gmsh.model.mesh.generate(2)
#ov = gmsh.model.geo.extrude([(2,s1),(2,s2),(2,s3)],0,0,gaugeThickness,[8],recombine=True)
ov = gmsh.model.geo.extrude([(2,s1),(2,s2)],0,0,gaugeThickness,[1],recombine=True)
gmsh.model.geo.synchronize()

gmsh.option.setNumber("Mesh.ElementOrder", 1)
#gmsh.option.setNumber("Mesh.HighOrderOptimize", 2)
#gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)

gmsh.model.mesh.generate(3)

gmsh.model.geo.synchronize()
delta = 5E-2
top_sur = gmsh.model.getEntitiesInBoundingBox(-gaugeWidth-5-delta,gaugeHeight+2-delta,-gaugeThickness-delta,
                                              gaugeWidth+5+delta, gaugeHeight+delta+2,gaugeThickness+delta,
                                              dim=2)

btm_sur = gmsh.model.getEntitiesInBoundingBox(-100-delta,0-delta,0-delta,
                                              100+delta,0+delta,gaugeThickness+delta,
                                              dim=2)

bck_sur = gmsh.model.getEntitiesInBoundingBox(-100,0-delta,0-delta,
                                              100,gaugeHeight+2+delta,0+delta,
                                              dim=2)

vis_sur = gmsh.model.getEntitiesInBoundingBox(-100,0-delta,gaugeThickness-delta,
                                               100,gaugeHeight+delta,gaugeThickness+delta,
                                               dim=2)

vol = gmsh.model.getEntitiesInBoundingBox(-100,-100,-100,
                                        100,100,100,
                                          dim=3)

gmsh.model.geo.addPhysicalGroup(3, [x[1] for x in vol], name='Specimen')
gmsh.model.geo.addPhysicalGroup(2, [x[1] for x in top_sur], name='Top-BC')
gmsh.model.geo.addPhysicalGroup(2, [x[1] for x in btm_sur], name='Y-Symm')
gmsh.model.geo.addPhysicalGroup(2, [x[1] for x in bck_sur], name='Z-Symm')
gmsh.model.geo.addPhysicalGroup(2, [x[1] for x in vis_sur], name='Visible-Surface')

gmsh.model.geo.synchronize()
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

exportpath = 'examples/mesh/circ.msh'
gmsh.write(exportpath)
gmsh.finalize()
