-- SD Syntax Highlighting Filter for Quarto/Pandoc
-- Uses standalone highlight script that only depends on lark

-- Get the directory where this filter is located
local function script_path()
  local str = debug.getinfo(2, "S").source:sub(2)
  return str:match("(.*/)")
end

local FILTER_DIR = script_path()

function CodeBlock(block)
  -- only process blocks with "sd" class
  if not block.classes:includes("sd") then
    return nil
  end

  -- write code to temp file
  local tmp_in = os.tmpname()
  local f = io.open(tmp_in, "w")
  f:write(block.text)
  f:close()

  -- call the standalone highlight script
  local script = FILTER_DIR .. "highlight_fragment.py"
  local cmd = "python3 '" .. script .. "' '" .. tmp_in .. "'"

  local handle = io.popen(cmd)
  local highlighted = handle:read("*all")
  handle:close()
  os.remove(tmp_in)

  if highlighted and #highlighted > 0 then
    local html = '<pre class="sd-code">' .. highlighted .. '</pre>'
    return pandoc.RawBlock("html", html)
  end

  -- fallback: return unchanged
  return nil
end
