#  @license
#  Copyright 2019 Google LLC. All Rights Reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Helper definitions to glob .aar and .jar targets"""

def create_aar_targets(aarfiles):
    for aarfile in aarfiles:
        name = "aars__" + aarfile[aarfile.rindex("/") + 1:aarfile.rindex(".aar")]
        lib_deps.append(":" + name)
        android_prebuilt_aar(
            name = name,
            aar = aarfile,
        )

def create_jar_targets(jarfiles):
    for jarfile in jarfiles:
        name = "jars__" + jarfile[jarfile.rindex("/") + 1:jarfile.rindex(".jar")]
        lib_deps.append(":" + name)
        prebuilt_jar(
            name = name,
            binary_jar = jarfile,
        )
