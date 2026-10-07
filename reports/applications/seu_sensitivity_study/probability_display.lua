function Table(element)
  local probability_column = nil
  for _, row in ipairs(element.head.rows) do
    for column, cell in ipairs(row.cells) do
      if pandoc.utils.stringify(cell.contents) == "Neutral probabilities" then
        probability_column = column
      end
    end
  end
  if probability_column == nil then
    return nil
  end
  for _, body in ipairs(element.bodies) do
    for _, row in ipairs(body.body) do
      local cell = row.cells[probability_column]
      local values = pandoc.json.decode(pandoc.utils.stringify(cell.contents))
      assert(type(values) == "table" and #values == 3, "Expected three probabilities")
      local formatted = {}
      for _, value in ipairs(values) do
        assert(type(value) == "number" and value >= 0 and value <= 1,
          "Expected finite probabilities in [0, 1]")
        table.insert(formatted, string.format("%.6f", value))
      end
      cell.contents = {pandoc.Plain({pandoc.Str("[" .. table.concat(formatted, ", ") .. "]")})}
    end
  end
  return element
end