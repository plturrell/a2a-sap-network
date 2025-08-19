import { Command } from 'commander';
import chalk from 'chalk';
import { initCommand } from './commands/init';
import { createCommand } from './commands/create';
import { devCommand } from './commands/dev';
import { configCommand } from './commands/config';
import { doctorCommand } from './commands/doctor';

const program = new Command();

// ASCII art logo
const logo = chalk.cyan(`
    ___   ___      ___   
   / _ | |_  |    / _ |  
  / __ | / __/   / __ |  
 /_/ |_|/____/  /_/ |_|  
                         
`);

program
  .name('a2a')
  .description('A2A Framework CLI - Build decentralized agents with ease')
  .version('1.0.0')
  .addHelpText('before', logo);

// Register commands
program.addCommand(initCommand);
program.addCommand(createCommand);
program.addCommand(devCommand);
program.addCommand(configCommand);
program.addCommand(doctorCommand);

// Parse command line arguments
program.parse(process.argv);

// Show help if no command provided
if (!process.argv.slice(2).length) {
  program.outputHelp();
}