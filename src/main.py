from run_experiments import *

if __name__ == "__main__":
    try:
        config = init()
        if len(sys.argv) > 2 and sys.argv[2]=='--dry-run':
            xs = list(build_experiments(config))
            print(f"prepared {len(xs)} experiments")
            if len(sys.argv) > 3 and sys.argv[3] == '--verbose':
                print()
                for r, conf in xs:
                    print(r.run_name,'\t',conf)
                print()
        else:
            paths = main(config)
            done(*paths)
    except UserFacingError as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Experiment batch cancelled by user. Shutting down gracefully.")
        sys.exit(130) # 130 is the standard exit code for Ctrl+C
    except Exception as e:
        logging.error(f"Experiment failed with a critical error: {e}", exc_info=True)
        raise

