//
//  PoissonReconTests.m
//  PoissonReconTests
//
//  Created by Aaron Thompson on 4/26/18.
//  Copyright © 2018 Standard Cyborg. All rights reserved.
//

#import <XCTest/XCTest.h>
#import <PoissonRecon/PoissonRecon.h>

@interface PoissonReconTests : XCTestCase

@end

@implementation PoissonReconTests

- (void)testPoissonRecon {
    NSString *inputPath = [[NSBundle bundleForClass:[self class]] pathForResource:@"app_scan" ofType:@"ply"];
    NSString *outputPath = [NSTemporaryDirectory() stringByAppendingPathComponent:@"output.ply"];
    
    NSLog(@"Running Poisson on file %@", inputPath);
    
    PoissonReconOperation *operation = [[PoissonReconOperation alloc] initWithInputFilePath:inputPath outputFilePath:outputPath];
    [operation start];
    
    NSLog(@"Finished with output at %@", outputPath);
}

@end
