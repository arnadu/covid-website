
import * as ec2 from "@aws-cdk/aws-ec2";
import * as ecs from "@aws-cdk/aws-ecs";
import * as ecs_patterns from "@aws-cdk/aws-ecs-patterns";

import * as cdk from '@aws-cdk/core';
import * as r53 from '@aws-cdk/aws-route53'
import * as path from 'path';

import Duration from '@aws-cdk/core';

//import { App, Construct } from '@aws-cdk/core';

export class InfrastructureStack extends cdk.Stack {
  constructor(scope: cdk.Construct, id: string, props?: cdk.StackProps) {

    super(scope, id, props);

    // The code that defines your stack goes here
    const vpc = new ec2.Vpc(this, "MyVpc", {
      maxAzs: 3 // Default is all AZs in region
    });

    const cluster = new ecs.Cluster(this, "MyCluster", {
      vpc: vpc
    });

    const zone = r53.HostedZone.fromLookup(this, 'Zone', { domainName: 'arnadu.com' });

    // Create a load-balanced Fargate service and make it public
    const service = new ecs_patterns.ApplicationLoadBalancedFargateService(this, "MyFargateService", {
      cluster: cluster, // Required
      cpu: 256, // Default is 256
      desiredCount: 1, // Default is 1
      taskImageOptions: { image: ecs.ContainerImage.fromAsset("../covid-website"),
                          containerPort: 5006},
      memoryLimitMiB: 2048, // Default is 512
      publicLoadBalancer: true, // Default is false
      domainName: "covid-calculator.arnadu.com",
      domainZone: zone
    });
    
    service.targetGroup.enableCookieStickiness(Duration.minutes(60));
    
  }
}

