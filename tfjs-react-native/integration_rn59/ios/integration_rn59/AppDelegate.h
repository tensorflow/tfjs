/**
 * Copyright (c) Facebook, LLC and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import <React/RCTBridgeDelegate.h>
#import <UIKit/UIKit.h>
#import <UMReactNativeAdapter/UMModuleRegistryAdapter.h>

@interface AppDelegate : UIResponder <UIApplicationDelegate, RCTBridgeDelegate>

@property(nonatomic, strong) UMModuleRegistryAdapter *moduleRegistryAdapter;
@property(nonatomic, strong) UIWindow *window;

@end
