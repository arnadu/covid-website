#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from '@aws-cdk/core';
import { InfrastructureStack } from '../lib/infrastructure-stack';

//const envDebug = { account: '833015572711', region: 'us-east-1' };

const envDebug = { 
    account: process.env.CDK_DEFAULT_ACCOUNT, 
    region: process.env.CDK_DEFAULT_REGION
};


const app = new cdk.App();
new InfrastructureStack(app, 'InfrastructureStack', { env: envDebug } );
