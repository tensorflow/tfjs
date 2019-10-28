package com.integration_rn59.generated;

import java.util.Arrays;
import java.util.List;
import org.unimodules.core.interfaces.Package;

public class BasePackageList {
  public List<Package> getPackageList() {
    return Arrays.<Package>asList(
        new expo.modules.camera.CameraPackage(),
        new expo.modules.constants.ConstantsPackage(),
        new expo.modules.filesystem.FileSystemPackage(),
        new expo.modules.gl.GLPackage(),
        new expo.modules.imagemanipulator.ImageManipulatorPackage(),
        new expo.modules.permissions.PermissionsPackage()
    );
  }
}
