//
//  PoissonReconOperation.mm
//  PoissonRecon
//
//  Created by Aaron Thompson on 4/26/18.
//  Copyright © 2018 Standard Cyborg. All rights reserved.
//

#import "PoissonReconOperation.h"
#import "PoissonReconExecute.h"

using namespace std;

@implementation PoissonReconOperation {
    NSString *_inputFilePath;
    NSString *_outputFilePath;
}

- (instancetype)initWithInputFilePath:(NSString *)inputPath
                       outputFilePath:(NSString *)outputPath
{
    self = [super init];
    if (self) {
        _inputFilePath = inputPath;
        _outputFilePath = outputPath;
    }
    return self;
}

- (void)main
{
    Execute(_inputFilePath, _outputFilePath);
}

@end
