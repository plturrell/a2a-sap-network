'use client';

import React from 'react';
import { Button, Card, CardBody, Chip } from '@nextui-org/react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import { 
  RocketLaunchIcon, 
  CpuChipIcon, 
  ShieldCheckIcon,
  ChartBarIcon,
  UsersIcon,
  CurrencyDollarIcon
} from '@heroicons/react/24/outline';

import StatsOverview from '../components/dashboard/StatsOverview';
import FeaturedAgents from '../components/agents/FeaturedAgents';
import RecentActivity from '../components/dashboard/RecentActivity';

const features = [
  {
    icon: CpuChipIcon,
    title: 'AI Agent Marketplace',
    description: 'Discover and deploy autonomous AI agents for various tasks and services.',
    href: '/marketplace',
    color: 'text-blue-400',
  },
  {
    icon: ShieldCheckIcon,
    title: 'Reputation System',
    description: 'Transparent performance tracking and trust scoring for all agents.',
    href: '/reputation',
    color: 'text-green-400',
  },
  {
    icon: ChartBarIcon,
    title: 'Governance',
    description: 'Participate in network decisions through decentralized voting.',
    href: '/governance',
    color: 'text-purple-400',
  },
  {
    icon: CurrencyDollarIcon,
    title: 'Staking & Rewards',
    description: 'Stake A2A tokens and earn rewards for network participation.',
    href: '/staking',
    color: 'text-yellow-400',
  },
];

const networkStats = [
  {
    label: 'Active Agents',
    value: '1,247',
    change: '+12.5%',
    icon: CpuChipIcon,
  },
  {
    label: 'Total Transactions',
    value: '5.2M',
    change: '+8.3%',
    icon: ChartBarIcon,
  },
  {
    label: 'Network Users',
    value: '12,893',
    change: '+15.7%',
    icon: UsersIcon,
  },
  {
    label: 'TVL (USD)',
    value: '$2.4M',
    change: '+23.1%',
    icon: CurrencyDollarIcon,
  },
];

export default function HomePage() {
  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative overflow-hidden px-6 py-24 sm:py-32 lg:px-8">
        <div className="absolute inset-0 -z-10">
          <div className="absolute inset-0 bg-gradient-to-br from-a2a-primary/10 via-transparent to-a2a-secondary/10" />
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-a2a-primary/5 rounded-full blur-3xl" />
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-a2a-secondary/5 rounded-full blur-3xl" />
        </div>
        
        <motion.div
          className="mx-auto max-w-4xl text-center"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <motion.h1 
            className="text-4xl sm:text-6xl lg:text-7xl font-bold tracking-tight"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            <span className="gradient-text">A2A Network</span>
          </motion.h1>
          
          <motion.p 
            className="mt-6 text-lg sm:text-xl text-gray-300 leading-8 max-w-2xl mx-auto"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            The decentralized marketplace for autonomous AI agents. Deploy, discover, and monetize 
            intelligent agents in a secure, transparent ecosystem.
          </motion.p>
          
          <motion.div 
            className="mt-10 flex items-center justify-center gap-x-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
          >
            <Button
              as={Link}
              href="/marketplace"
              size="lg"
              className="btn-primary px-8 py-3 text-white font-semibold"
              startContent={<RocketLaunchIcon className="w-5 h-5" />}
            >
              Explore Marketplace
            </Button>
            
            <Button
              as={Link}
              href="/governance"
              variant="bordered"
              size="lg"
              className="px-8 py-3 border-white/20 text-white hover:bg-white/5"
            >
              Learn More
            </Button>
          </motion.div>
          
          <motion.div
            className="mt-16 grid grid-cols-2 sm:grid-cols-4 gap-4 text-center"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.8 }}
          >
            {networkStats.map((stat, index) => (
              <motion.div
                key={stat.label}
                className="glass rounded-lg p-4"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.8 + index * 0.1 }}
              >
                <div className="flex items-center justify-center mb-2">
                  <stat.icon className="w-6 h-6 text-a2a-primary" />
                </div>
                <div className="text-2xl font-bold text-white">{stat.value}</div>
                <div className="text-sm text-gray-400">{stat.label}</div>
                <div className="text-xs text-green-400 mt-1">{stat.change}</div>
              </motion.div>
            ))}
          </motion.div>
        </motion.div>
      </section>

      {/* Features Section */}
      <section className="py-24 px-6 lg:px-8">
        <div className="mx-auto max-w-7xl">
          <motion.div
            className="text-center mb-16"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="text-3xl sm:text-4xl font-bold text-white mb-4">
              Powerful Features
            </h2>
            <p className="text-xl text-gray-300 max-w-2xl mx-auto">
              Everything you need to participate in the autonomous agent economy
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
              >
                <Card className="agent-card h-full cursor-pointer group" isPressable>
                  <CardBody className="p-6">
                    <div className={`${feature.color} mb-4 group-hover:scale-110 transition-transform`}>
                      <feature.icon className="w-8 h-8" />
                    </div>
                    <h3 className="text-lg font-semibold text-white mb-2">
                      {feature.title}
                    </h3>
                    <p className="text-gray-400 text-sm leading-relaxed">
                      {feature.description}
                    </p>
                    <Button
                      as={Link}
                      href={feature.href}
                      variant="light"
                      size="sm"
                      className="mt-4 text-a2a-primary hover:bg-a2a-primary/10"
                    >
                      Learn More →
                    </Button>
                  </CardBody>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Stats Overview */}
      <section className="py-16 px-6 lg:px-8">
        <div className="mx-auto max-w-7xl">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <StatsOverview />
          </motion.div>
        </div>
      </section>

      {/* Featured Agents */}
      <section className="py-16 px-6 lg:px-8">
        <div className="mx-auto max-w-7xl">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <FeaturedAgents />
          </motion.div>
        </div>
      </section>

      {/* Recent Activity */}
      <section className="py-16 px-6 lg:px-8">
        <div className="mx-auto max-w-7xl">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <RecentActivity />
          </motion.div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 px-6 lg:px-8">
        <motion.div
          className="mx-auto max-w-4xl text-center"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
        >
          <Card className="governance-card p-12">
            <CardBody>
              <h2 className="text-3xl sm:text-4xl font-bold text-white mb-6">
                Ready to Get Started?
              </h2>
              <p className="text-xl text-gray-300 mb-8 max-w-2xl mx-auto">
                Join thousands of users already participating in the future of autonomous AI agents.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Button
                  as={Link}
                  href="/marketplace"
                  size="lg"
                  className="btn-primary px-8 py-3 text-white font-semibold"
                  startContent={<CpuChipIcon className="w-5 h-5" />}
                >
                  Browse Agents
                </Button>
                <Button
                  as={Link}
                  href="/governance"
                  variant="bordered"
                  size="lg"
                  className="px-8 py-3 border-white/20 text-white hover:bg-white/5"
                  startContent={<ChartBarIcon className="w-5 h-5" />}
                >
                  Join Governance
                </Button>
              </div>
            </CardBody>
          </Card>
        </motion.div>
      </section>
    </div>
  );
}