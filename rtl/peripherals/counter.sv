module counter(
    input logic         clk, rst, 
    output logic [3:0]  count,
    output logic        flipper
);

    flip flip_inst(
        .clk(clk),
        .rst(rst),
        .flipper(flipper)
    );
    
    always_ff @(posedge clk) begin
        if (rst) begin
            count <= 4'd0;
        end else begin
            count <= count + 4'd1; 
        end
    end

endmodule