import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add src to python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from reconstruction.text_reconstruction import IterativeReconstructionStrategy
from llm.local_llm import HuggingFaceModelAdapter
from llm.prompting import ClozePromptBuilder
from data_models.captions_only import CaptionedVideo, CaptionedClip, TimestampRange

def verify_refactor():
    print("Verifying Refactor Integration...")

    # 1. Setup Logic
    # Mock the adapter so we don't load GPU models
    mock_adapter = MagicMock()
    # Mocking noisy output to test cleaning
    mock_adapter.call.return_value = '  "[00:00] A cat jumps over the fence."  '
    
    # Load Real Prompt Builder
    prompts_dir = Path(__file__).parent.parent / "prompts" / "iterative_cloze"
    if not prompts_dir.exists():
        print(f"FAIL: Prompt dir not found at {prompts_dir}")
        return

    prompt_builder = ClozePromptBuilder.from_directory(prompts_dir)
    
    # Init Strategy
    strategy = IterativeReconstructionStrategy(
        name="TestIterative",
        model_adapter=mock_adapter,
        prompt_builder=prompt_builder,
        config={"type": "local_llm", "max_new_tokens": 50}
    )

    # 2. Setup Data
    # 3 Clips: Present, Masked, Present
    clips = [
        CaptionedClip(index=0, timestamp=TimestampRange(start=0.0, duration=2.0), caption=None), # Masked Start
        CaptionedClip(index=1, timestamp=TimestampRange(start=2.0, duration=2.0), caption="Second caption."),
        CaptionedClip(index=2, timestamp=TimestampRange(start=4.0, duration=2.0), caption="Third caption.")
    ]
    video = CaptionedVideo(video_id="test_vid", clips=clips)

    # 3. Compile
    print("Running reconstruction...")
    result = strategy.reconstruct(video)

    # 4. Verify
    print(f"Result keys: {result.reconstructed_captions.keys()}")
    print(f"Result values: {result.reconstructed_captions.values()}")

    assert 0 in result.reconstructed_captions
    assert result.reconstructed_captions[0] == "A cat jumps over the fence."
    
    # Verify Adapter call
    mock_adapter.call.assert_called_once()
    call_args = mock_adapter.call.call_args
    
    kwargs = call_args[1]
    title_messages = kwargs.get('messages')
    if not title_messages:
        if call_args[0]:
            title_messages = call_args[0][0]
            
    messages = title_messages 
    user_content = messages[1]['content']
    print(f"Prompt Content:\n{user_content}")
    
    # Expectation: Context Before is invalid, uses start template
    # Template: "Task: Describe the opening scene at [{TARGET_TIMESTAMP}]."
    assert "Describe the opening scene" in user_content
    # Check for duration (2.0s from test data)
    assert "(duration: 2.0s)" in user_content
    # Context Before section should be GONE entirely in the new template
    assert "CONTEXT BEFORE" not in user_content 
 


if __name__ == "__main__":
    verify_refactor()
