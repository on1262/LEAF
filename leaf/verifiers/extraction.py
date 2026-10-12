import re
import io
import tokenize


def extract_last_boxed(text, code=False):
    """Return the final box, preserving nested braces and program formatting.

    An unfinished final box is invalid, even if an earlier box was complete.
    Braces escaped with an odd number of backslashes are literal TeX braces.
    """
    matches = list(re.finditer(r'\\boxed\s*\{', text))
    if not matches:
        return None
    start = matches[-1].end()
    if code:
        content = text[start:]
        fence = re.match(r'\s*```(?:python|py)?\s*\n', content, re.I)
        if fence:
            end = content.find('```', fence.end())
            if end < 0 or not re.match(r'\s*}', content[end+3:]):
                return None
            return content[:end+3].strip()
        offsets = [0]
        for line in content.splitlines(keepends=True):
            offsets.append(offsets[-1] + len(line))
        depth = 1
        try:
            for token in tokenize.generate_tokens(io.StringIO(content).readline):
                if token.type != tokenize.OP:
                    continue
                if token.string == '{':
                    depth += 1
                elif token.string == '}':
                    depth -= 1
                    if depth == 0:
                        return content[:offsets[token.start[0]-1]+token.start[1]].strip()
        except (tokenize.TokenError, IndentationError):
            pass
        return None
    depth = 1
    for i in range(start, len(text)):
        backslashes = 0
        j = i - 1
        while j >= 0 and text[j] == '\\':
            backslashes += 1; j -= 1
        if backslashes % 2:
            continue
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return None


def unwrap(text):
    text = str(text).strip().strip('$')
    for _ in range(8):
        match = re.match(r'^\\(?:text|textbf|mathrm|mathbf|bold)\{', text)
        if not match:
            break
        depth=1
        end=match.end()
        while end<len(text) and depth:
            if text[end]=='{':depth+=1
            elif text[end]=='}':depth-=1
            end+=1
        if depth or end!=len(text):
            break
        text = text[match.end():end-1].strip()
    return text
