""" Allow to associate a python function called following the 'EndEvent' coming from
a given ThreeD renderer managed by Slicer layout manager.
Usage:

  # Install screenshot grabber
      id = install_threed_view_action(takeScreenshot)

  # Move threeD view around

  # Uninstall screenshot grabber
  uninstall_threed_view_action(id)
"""

TAKE_SCREENSHOT_DEFAULT_PATH = '/tmp'
TAKE_SCREENSHOT_DEFAULT_PREFIX = 'screenshot_'


def takeScreenshot(rw, event):
    import os
    import datetime as dt
    ms_since_epoch = datetime.datetime.now().strftime('%s.%f')
    filename = rw.screenshot_prefix + str(ms_since_epoch) + '.png'
    filepath = os.path.join(rw.screenshot_path, filename)

    print("screenshot:%s" % filepath)  # Not implemented
    rw = view.renderWindow()
    wti = vtk.vtkWindowToImageFilter()
    wti.SetInput(rw)
    wti.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(filepath)
    writer.SetInputConnection(wti.GetOutputPort())
    writer.Write()


def threed_renderwindow(view_id=0):
    lm = slicer.app.layoutManager()
    viewWidget = lm.threeDWidget(view_id)
    view = viewWidget.threeDView()
    return view.renderWindow()


def install_threed_view_action(
        action, view_id=0,
        path=TAKE_SCREENSHOT_DEFAULT_PATH,
        prefix=TAKE_SCREENSHOT_DEFAULT_PREFIX):
    rw = threed_renderwindow(view_id)
    rw.screenshot_path = path
    rw.screenshot_prefix = prefix
    tag = rw.AddObserver('EndEvent', action)
    return tag


def uninstall_threed_view_action(action_id, view_id=0):
    rw = threed_renderwindow(view_id)
    rw.RemoveObserver(action_id)