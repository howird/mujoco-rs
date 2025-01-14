use clap::Parser;
use mujoco_visualiser::AppBuilder;

#[derive(Parser, Debug)]
#[command(version, about, long_about=None)]
struct Args {
    // XML file of the robot/actor to load
    #[arg(short, long)]
    xml: String,
}

fn main() {
    let args = Args::parse();

    let app = AppBuilder::from_xml(args.xml)
        .unwrap()
        .with_default_rendering()
        .build();

    // Run the simulator
    app.run_app();
}
